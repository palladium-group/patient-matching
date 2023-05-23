#!/usr/bin/env python
# coding: utf-8

# Used to monitor process using standard unix tools
import setproctitle
setproctitle.setproctitle('pm')

# Crude but fast terminal logging
import datetime 
llevels = ['INFO', 'STATUS', 'WARN', 'ERROR']
logg = lambda l, x: print(f'{datetime.datetime.now()}: [{llevels[l]}] {x}', flush=True)
logg(1, 'Starting')

# Record Linkage Toolkit is chatty by default, disable this
import logging 
logging.getLogger('recordlinkage').setLevel(logging.CRITICAL)

# Python Standard Library
import collections 
# import datetime 
import glob 
import itertools 
import json 
# import logging 
import multiprocessing 
import pathlib 
import pickle 
import re 
import shutil 
import sys 
import tempfile
import getopt 

# External (3rd Party) Packages
import bitstring
import numpy as np
import pandas as pd
# import psycopg2
import recordlinkage as rl
# import setproctitle
# import sklearn.ensemble
import sqlalchemy
import tqdm
import tqdm.contrib.concurrent

argv = sys.argv[1:]
pm_scripts_path = ''

try:
    opts, args = getopt.getopt(argv, 'p:', ['pm_scripts_path'])
    if len(opts) == 0 or len(opts) > 1:
        print('usage: -p <pm_scripts_path>')
        sys.exit(1)
    else:
        pm_scripts_path = opts[0][1]
except:
    logg(3, 'Error with options')
    sys.exit(1)

try:
    cache_dir = tempfile.mkdtemp()
    logg(0, f'Cache Directory: {cache_dir}')
except:
    logg(3, 'Could not create Cache Directory')
    sys.exit(1)

def psql_engine(c_file):
    try:
        c = json.load(open(c_file))
        s = f'postgresql+psycopg2://{c["user"]}:{c["password"]}@{c["host"]}/{c["dbname"]}'
        return sqlalchemy.create_engine(s, echo=False)
    except:
        pprint(3, 'Could not connect to Database')
        shutil.rmtree(cache_dir)
        sys.exit(1)

def load_data():
    engine = psql_engine(pm_scripts_path + '/credentials.json')

    logg(1, 'Querying Database')
    setproctitle.setproctitle('pm --database')
    
    # Only select columns actively in use
    columns = (
        'id',                   # Globally Unique
        'nk',                   # Globally Unique
        'id_number',           ## Matching String
        'first_name',          ## Matching String
        'last_name',           ## Matching String
        'date_of_birth',       ## Matching Date
        'gender',              ## Matching Boolean
    )

    data_columns = ('id',)
    date_columns = ('date_of_birth', )
    # categorical_columns = ('gender', )
    upper_columns = [f'UPPER({x}) as {x}' if (x not in data_columns and x not in date_columns) else x for x in columns]

    # Only select records from database that are active
    query = f'SELECT {", ".join(upper_columns)} FROM dim.patient;'
    
    try:
        patients = pd.read_sql(query, engine, index_col='id')
    except:
        logg(3, 'Could not query Database')
        shutil.rmtree(cache_dir)
        sys.exit(1)
    
    # Database nulls are 1/1/1 and 9999/12/31; Global ID requres dates between 1870 and 2125
    invalid_dates = (datetime.date(1, 1, 1), datetime.date(9999,12,31), )
    for column in date_columns:
        patients[column] = patients[column].map(lambda x: x if ((x not in invalid_dates) and \
                                                                (x.year <= 2125) and \
                                                                (x.year >= 1870)) \
                                                else pd.NaT).astype('datetime64[ns]')

    engine.dispose()
    
    return patients

patients = load_data()
try:
    patients.to_pickle(f'{cache_dir}/patients.pkl')
except:
    logg(3, 'Could not cache Patients')
    shutil.rmtree(cache_dir)
    sys.exit(1)

patients.reset_index(inplace=True)

# All IDs are bitshifted to permit multiple copies of the same ID in the pd Index
# IDs are shifted back before being converted to nk values
patients['id'] = patients['id'].apply(lambda x: x << 4)
patients.set_index('id', inplace=True)


logg(0, f'{patients.shape[0]} candidate patients')

logg(1, 'Calculating Exact Matches')
setproctitle.setproctitle('pm --exact')

# Exact matches are all pairs from pd Group
def calc_exact_match(g):
    c = pd.DataFrame(itertools.combinations(g, 2))
    try:
        c.to_csv(f'{cache_dir}/res_e_{multiprocessing.current_process().pid}.csv', header=False, index=False, mode='a')
    except:
        logg(3, 'Could not cache Exact Matches')
        shutil.rmtree(cache_dir)
        sys.exit(1)

    return c.shape[0]

# Don't consider rows with missing data for exact matches
dup_groups = patients.replace('', pd.NA).dropna(subset=['id_number', 'date_of_birth', 'gender'])
dup_groups = dup_groups[dup_groups.duplicated(['id_number', 'date_of_birth', 'gender'], keep=False)]\
    .groupby(['id_number', 'date_of_birth', 'gender']).groups.values()
    
exact_counts = tqdm.contrib.concurrent.process_map(calc_exact_match, dup_groups, chunksize=1, desc=str(datetime.datetime.now()))
logg(0, f'{sum(exact_counts)} exact matches')

del dup_groups
del exact_counts

# Only one of exact duplicates needs to be retained as others have already been captured as exact matches
logg(1, 'Removing Duplicate Patients')
setproctitle.setproctitle('pm --deduplicate')
patients.drop_duplicates(['id_number', 'first_name', 'last_name', 'date_of_birth', 'gender'], keep='first', inplace=True)
logg(0, f'{patients.shape[0]} candidate patients')

logg(1, 'Computing Blocking Metrics')
setproctitle.setproctitle('pm --blocking_metrics')
patients.reset_index(inplace=True)

# Return first letter of name, but consider only ASCII letters
def first_last(s):
    s = s.split()
    s = re.sub('[^A-Z]', '', s[0] if len(s) > 0 else '')
    if len(s) < 1:
        return '#'
    return s[0]

# Short Identifier for Gender
patients['gender_flp'] = patients['gender'].map(lambda x: x[0] if len(x) > 0 else '#')

# Short Identifier for Names
for column in ('first_name', 'last_name', ):
    patients[f'{column}_flp'] = patients[column].map(first_last)

# Number of names for multi-matching    
patients['fn_length'] = patients['first_name'].apply(lambda x : len(x.split()))

# Patient Year of Birth for Bucketing
patients['date_bucket'] = patients['date_of_birth'].map(lambda x: 9999 if pd.isnull(x) else x.year).astype('str')

# Compute Blocking Key from Above Short Identifiers and Year
patients['flp_index'] = patients['gender_flp'] + patients['first_name_flp'] + patients['last_name_flp'] + patients['date_bucket']

# In cases where there are multiple first names, the same record needs to be injected for each
# name, with an updated Blocking Key. The ID field needs to be made unique (incremented)
# based on the fact that it was previously bitshifted left.
def patient_generator():
    for patient in tqdm.tqdm(patients[patients['fn_length'] > 1].iterrows(), total=patients[patients['fn_length'] > 1].shape[0], desc=str(datetime.datetime.now())):
        patient = patient[1]
        first_names = patient['first_name'].split()
        local_flps = set()
        for first_name in first_names:
            fn_flp = first_last(first_name)
            flp_index = patient['gender_flp'] + fn_flp + patient['last_name_flp'] + patient['date_bucket']
            if flp_index not in local_flps:
                local_flps.add(flp_index)
                idf = patient['id'] + len(local_flps)
                yield (idf, patient['nk'], patient['id_number'], patient['first_name'], patient['last_name'], patient['date_of_birth'], patient['gender'], flp_index, )

patients = pd.concat([patients.loc[patients['fn_length'] <= 1, ['id', 'nk', 'id_number', 'first_name', 'last_name', 'date_of_birth', 'gender', 'flp_index']],\
                     pd.DataFrame(patient_generator(), columns=['id', 'nk', 'id_number', 'first_name', 'last_name', 'date_of_birth', 'gender', 'flp_index'])])
patients.set_index('id', inplace=True)

# Not safe to have more than ~2500 patients loaded per thread at once
# to prevent running out of memory. Figure determined experimentally
patients['bucket'] = '0'
bucket_size = 2500
index_sizes = patients.groupby('flp_index').size()

# Determine Blocking Indexes with >2500 records and break them into chunks of 2500
for group in tqdm.tqdm(index_sizes[round(index_sizes / bucket_size) > 1].index, desc=str(datetime.datetime.now())):
    pg_index = patients['flp_index'] == group
    local_size = int(round(patients[pg_index].size / bucket_size))
    patients.loc[pg_index, 'bucket'] =  pd.cut(patients[pg_index].index, local_size, labels=range(local_size)).astype('str')

# Produce and output some stats on the number of pairs per Blocking Key
# Stats can be used to monitor progress more accurately than using progress bar alone
# (as each unit of progress bar can be of a vastly different size)
index_sizes = index_sizes.reset_index()
index_sizes.columns = ['index', 'size']
index_sizes['pairs'] = ((index_sizes['size'] * index_sizes['size']) - index_sizes['size']) // 2
index_sizes['cumulative_pairs'] = index_sizes['pairs'].cumsum()
index_sizes['percentage'] = (index_sizes['pairs'] / sum(index_sizes['pairs'])) * 100
index_sizes['cumulative_percentage'] = index_sizes['percentage'].cumsum()
index_sizes.to_csv(f'{cache_dir}/index_sizes.csv', index=False, float_format='%2.3f')
    
logg(0, f'{patients.shape[0]} candidate patients')
logg(0, f'Classification Index Sizes: {cache_dir}/index_sizes.csv')

del bucket_size
del index_sizes

logg(1, 'Caching Data')
setproctitle.setproctitle('pm --cache')

# Cache each Blocking Index chunk of <=2500 to disk
groups = patients.groupby(['flp_index', 'bucket'])#.groups
groups_dict = groups.groups

for x in tqdm.tqdm(groups_dict.items(), desc=str(datetime.datetime.now())):
    try:
        f, i = x
        pathlib.Path(f'{cache_dir}/patients/{f[0]}/').mkdir(parents=True, exist_ok=True)
        patients.loc[i].to_pickle(f'{cache_dir}/patients/{f[0]}/patients_{f[0]}_{f[1]}.pkl')
    except:
        logg(3, 'Could not cache Chunk')
        shutil.rmtree(cache_dir)
        sys.exit(1)

# Also store a list of the Blocking Indexes
idx = groups.size().reset_index(name='counts')
idx.sort_values('counts', ascending=False, inplace=True)
gps = idx.groupby('flp_index').groups

keys = pd.Series(list(gps.keys()))
try:    
    keys.to_pickle(f'{cache_dir}/gps_keys.pkl')
except:
    logg(3, 'Could not cache Blocking Index')
    shutil.rmtree(cache_dir)
    sys.exit(1)

# And an index, per Blocking Key, of chunks in that key
pathlib.Path(f'{cache_dir}/datasets/').mkdir(parents=True, exist_ok=True)

for flp in tqdm.tqdm(keys, desc=str(datetime.datetime.now())):
    group = gps[flp]
    dataset = idx['bucket'].loc[group]
    try:
        dataset.to_pickle(f'{cache_dir}/datasets/dataset_{flp}.pkl')
    except:
        logg(3, 'Could not cache Chunk Index')
        shutil.rmtree(cache_dir)
        sys.exit(1)

del patients
del groups
del groups_dict
del idx
del gps
del keys

# Features for Record Linkage comparison
comp = rl.Compare(n_jobs=1, indexing_type='position')
comp.string('first_name', 'first_name', method='jarowinkler', label='first_name')
comp.string('last_name', 'last_name', method='jarowinkler', label='last_name')
comp.date('date_of_birth', 'date_of_birth', label='date_of_birth')


logg(1, 'Classifying')
setproctitle.setproctitle('pm --classify')

# Load pregenerated sklearn model
with open(pm_scripts_path + '/clf.pkl', 'rb') as f:
    et_clf = pickle.load(f)

# Classify and Cache Individual Blocking Index Value    
def single_flp_group(flp):
    setproctitle.setproctitle(f'pm --classify --block={flp.lower()}')
    
    try:
        datasets = pd.read_pickle(f'{cache_dir}/datasets/dataset_{flp}.pkl')
        
        total_candidate = 0
        total_match = 0
        
        # Enumerate over all unique pairs of chunks; here called datasets
        for i, dataset_1 in enumerate(datasets):
            for dataset_2 in datasets[i:]:
                # Case where chunk pairs are both the same chunk
                if dataset_1 == dataset_2:
                    x = pd.read_pickle(f'{cache_dir}/patients/{flp}/patients_{flp}_{dataset_1}.pkl')
                    li = rl.index.Full().index(x) # Full Index because chunk is already blocked
                    li_size = li.size
                    total_candidate += li_size
                    if li_size > 0: # Chunks with only one record won't be processed
                        p = et_clf.predict_proba(comp.compute(li, x))
                        x = np.where(p[:,1] >= 0.9) # Only accept records with probability of >=0.9
                        li = li[x]
                        li_size = li.size
                        if li_size > 0: # Only store if there were matches
                            total_match += li_size
                            li.to_frame().to_csv(f'{cache_dir}/res_{multiprocessing.current_process().pid}.csv', header=False, index=False, mode='a')

                # Case where the chunk pair is different
                # Code almost identical to above
                else:
                    x = pd.read_pickle(f'{cache_dir}/patients/{flp}/patients_{flp}_{dataset_1}.pkl')
                    y = pd.read_pickle(f'{cache_dir}/patients/{flp}/patients_{flp}_{dataset_2}.pkl')
                    li = rl.index.Full().index(x, y)
                    li_size = li.size
                    total_candidate += li_size
                    if li_size > 0:
                        p = et_clf.predict_proba(comp.compute(li, x, y))
                        x = np.where(p[:,1] >= 0.9)
                        li = li[x]
                        li_size = li.size
                        if li_size > 0:
                            total_match += li_size
                            li.to_frame().to_csv(f'{cache_dir}/res_{multiprocessing.current_process().pid}.csv', header=False, index=False, mode='a')

        # Return values are statistics; actual result is cached to disk
        return (total_candidate, total_match)
    
    except Exception as e:
        logg(3, f'Could not classify {flp}')
        return (0 ,0)

sl = tqdm.contrib.concurrent.process_map(single_flp_group, pd.read_pickle(f'{cache_dir}/gps_keys.pkl'), chunksize=1, desc=str(datetime.datetime.now()))

del et_clf
del comp

# Calculate a few stats from classification
cp, mp = [sum(x) for x in zip(*sl)]
logg(0, f'{cp} total pairs; {mp} matched pairs')

del sl
del cp
del mp

logg(1, "Assigning Groups")
setproctitle.setproctitle('pm --group')
group_ids = dict()
group_key = -1
for fn in tqdm.tqdm(glob.glob(f'{cache_dir}/res_*.csv'), desc=str(datetime.datetime.now())):
    index = pd.read_csv(fn, names=['id1', 'id2'], header=None)
    
    for row in index.iterrows():
        id1, id2 = row[1]
        # Reverse bitshift on ID
        id1 = id1 >> 4
        id2 = id2 >> 4

        # Ignore cases where record matched itself
        # this should happen with multiple name records
        if id1 == id2:
            continue
            
        id1, id2 = sorted([id1, id2])

        # Neither records seen before, generate new group_id
        if id1 not in group_ids and id2 not in group_ids:
            group_key += 1
            group_ids[id1] = group_key
            group_ids[id2] = group_key
        # One record seen before, reuse other group_id
        elif id1 in group_ids and id2 not in group_ids:
            group_ids[id2] = group_ids[id1]
        # Opposite seen before, reuse other group_id
        elif id1 not in group_ids and id2 in group_ids:
            group_ids[id1] = group_ids[id2]
        # Seen both previously
        else: # if s in group_ids and l in group_ids
            # Reindex second group into first group if both are in different groups
            if group_ids[id1] != group_ids[id2]:
                for k, v in group_ids.items():
                    if v == group_ids[id2]:
                        group_ids[k] = group_ids[id1]

logg(0, f'{len(group_ids)} total patients; {group_key + 1} total groups')
try:
    pd.DataFrame(group_ids.items(), columns=['id', 'group_id']).set_index('id').to_pickle(f'{cache_dir}/group_ids.pkl')
except:
    logg(3, 'Could not cache Group Assignments')
    shutil.rmtree(cache_dir)
    sys.exit(1)

del group_ids
del group_key

patients = pd.read_pickle(f'{cache_dir}/patients.pkl')
groups = pd.read_pickle(f'{cache_dir}/group_ids.pkl')

patients = patients.join(groups, how='inner').reset_index().sort_values(by=['date_of_birth', 'id'])

logg(1, "Computing Universal Identifiers")
setproctitle.setproctitle('pm --universal')

sequences = collections.Counter()
mif = 'uint:58=pif, uint:5=sequence, pad:1=parity'

pif = '''
    uint:8=year,
    uint:4=month,
    uint:5=day,
    bool=gender,
    bits:20=last_name,
    bits:20=first_name,
    '''

nif = 'uint:5, uint:5, uint:5, uint:5'

re_sub = re.compile(r'[^A-Z]')

def encode_name(name):
    try:
        # Encode 4 characters of name into a numeric value based on ordinal position. Consider only ASCII letters.
    	return bitstring.pack(nif, *([0] * 4 + [ord(x) - 64 for x in re_sub.sub(r'', name.upper())[:4]])[-4:])
    except:
        logg(2, 'Failed to encode name for Unique Identifier, assigning 0')
        return bitstring.BitArray(20)

def encode_record(r):
    try:
        year = max(r['date_of_birth'].year - 1870, 0)
    
        bs = bitstring.pack(pif,
                   year=year,
                   month=r['date_of_birth'].month,
                   day=r['date_of_birth'].day,
                   gender=(r['gender'] == 'MALE'),
                   last_name=encode_name(r['last_name']),
                   first_name=encode_name(r['first_name']))
    
        return bs.uint
    except:
        logg(2, 'Failed to produce Unique Identifier, assigning 0')
        return 0

# Unique Identifier has a sequence value, compute and assign based on rest of Identifier
def encode_sequence(r):
    try:
        r = patients.loc[r]
        pif = encode_record(r)
        sequences[pif] += 1
        bs = bitstring.pack(mif,
                           pif=pif,
                           sequence=sequences[pif])
        bs[-1] = bs.count(1) % 2 # Simplistic check bit
        
        return (r['group_id'], bs.uint, )
    except:
        logg(2, 'Failed to Sequence Unique Identifier, assigning 0')
        return (r['group_id'], 0, )


dup_ids = patients.duplicated('group_id', keep='first')
# Only compute Universal ID once per group
universal_ids = { k: v for k, v in tqdm.contrib.concurrent.process_map(encode_sequence, patients[~dup_ids].index, chunksize=1, desc=str(datetime.datetime.now())) }
patients['group_id'] = patients['group_id'].map(universal_ids).astype('uint64')

# patients.sort_values(['group_id', 'nk'], inplace=True)
# patients[['nk', 'group_id']].to_csv("results.csv", index=False)

del dup_ids
del universal_ids
del sequences

logg(1, "Writing to Database")
setproctitle.setproctitle('pm --db_write')

engine = psql_engine(pm_scripts_path + '/credentials.json')

sql_statement = '''INSERT INTO mlm.patient_matches (nk, group_id, created, updated)
VALUES (%(nk)s, %(group_id)s, %(created)s, %(updated)s)
ON CONFLICT (nk)
DO UPDATE SET group_id = EXCLUDED.group_id, updated = EXCLUDED.updated;'''

# Records are upserted using Postgres ON CONFLICT, and in the udate case, 'created' is not written
time_stamp = datetime.datetime.now()
patients['created'] = time_stamp
patients['updated'] = time_stamp

try:
    if patients.shape[0] > 0:
        engine.execute(sql_statement, patients[['nk', 'group_id', 'created', 'updated']].to_dict(orient='records'))
except:
    logg(3, 'Could not write results to Database')
    shutil.rmtree(cache_dir)
    sys.exit(1)

engine.dispose()

logg(1, 'Cleaning Up')
setproctitle.setproctitle('pm --clean')
shutil.rmtree(cache_dir)

logg(1, 'Done')
setproctitle.setproctitle('pm')
