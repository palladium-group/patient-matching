## Installation and Use
This section details the installation and use of the patient matching solution. 
The solution is largely platform agnostic, requiring only a working version of Python 3 and the appropriate libraries (as discussed in the section on implementation) to function. As Microsoft Windows and Python 3 can be difficult to integrate, this solution is not tested on Windows and may well not function under that environment. Functionally, the system should, however work without modification under the Windows Subsystem for Linux (WSL).

### Prerequisites
The patient matching system is designed to run as a self-contained entity on most Linux systems and requires little pre-requisite software. It is dependent on a Python 3 runtime and several third-party Python libraries.
Before the tool can be installed and configured, several prerequisites must be met:
1.	The python3, python3-pip and python3-venv system packages must be installed. On some OS configurations, the python3-venv package may be built into the python3 package and then is not required separately. Note that an operating system with at least Python 3.6.8 is required, but Python 3.7 or newer is recommended.
On Debian machines these tools can be installed with apt install and on Centos machines, they can be installed with yum install.

2.	A full compiler toolchain, including GNU make, GCC and other tools is required. These can be installed using a system package manager.

    On Debian machines this can be installed as follows:

    `apt install build-essential`

    `apt install python3-dev`

    On Centos based systems, this can be automated as follows:
        
    `yum groupinstall Development Tools`

    `yum install python3-devel`

###  Configuring Python
This repository consists of Python files forming the patient matching system, a Makefile to automate common execution tasks and a configuration file in JSON format.
It is recommended that the Python dependencies are installed using a Python virtual environment. 
This virtual environment can live anywhere on the system, but it is suggested to keep it in the same folder as the patient matching code. Note that a virtual environment is not portable, if the installation is moved, it must be deleted and recreated.
The creation of the virtual environment can be automated using the provided make file:

`make create_environment`

or can be created manually:
	
`python3 -m venv venv`

If the provided Makefile is used at any stage, the virtual environment must be named venv and must live in the same directory as the Makefile.
All remaining tasks (including running the patient matching system) must be run in the context of the virtual environment. This activation happens automatically if using the Makefile but must otherwise be activated manually. This can be done by running:

`source venv/bin/activate`

It takes the path to the activation script, so if the user is not in the installation directory, the path should be adapted. On most shells, the prompt will be updated to reflect the activation.

### Starting the Installation
The Python script is dependent on several Python libraries as listed in the file requirements.txt. These libraries must be installed into the virtual environment from the Python Package Index.
The Python virtual environment requires some prerequisite packages to be installed or upgraded before these dependencies can be installed. To install and upgrade these prerequisites run:

`python3 -m pip install -U pip setuptools wheel`

Once these packages have been installed, the prerequisite packages can be installed using:

`python3 -m pip install -r requirements.txt`

Both these steps can be automated using the make target:

`make requirements`

### Configuring Patient Matching
The patient matching collects patient data from a Postgres database and writes its results back to the same database. Database credentials are read from the file credentials.json which is in JSON format. The file is as follows:
	
    {
		"dbname":   "",
		"user":     "",
		"password": "",
		"host":     ""
	}
    
These four fields should be completed in accordance with the database that the system is to be run against. The database must have two tables: *dim.patient* and *mlm.patient_matching* for the patient matching to run successfully. As this file contains a plaintext database password, permissions on the file should be set securely to prevent unauthorized access.

### Running the Patient Matching
Once all prerequisites above have been met, the full patient matching can be run, by using the following command:

`python3 patient_matching.py`

This command logs output to both stdout and stderr (the Terminal) so progress can be monitored. If this script is run interactively, it should be run in screen or tmux as it will take a significant amount of time to run.
Note that a progress log is written to stdout and progress bars for long running tasks are written to stderr. For use in an automated environment, these commands should be redirected appropriately.
This run of the patient matching can also be run via the Makefile, where the environment will be automatically setup, dependencies installed and the matching run. This can be done with the following make target:

`make patient_matching`

