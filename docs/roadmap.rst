Qcodes project plan
===================


Phase 1
-------
Initial release to tech-savvy experimenters.

-  Python 3 framework
-  Optimized for jupyter notebook, but compatible with terminal shell or
   standalone scripts run from the command line
-  Developed in this github repository (qdev-dk/Qcodes) and all code
   changes are reviewed
-  Modular architecture:
-  Instrument drivers can be easily plugged in
-  Components can be used or not used per each user's preferences
-  Other functions such as visualization packages, data storage back
   ends, and note-taking capabilities are also easy to swap in or out.
-  Core package, configuration, and experiment scripts are all separate
-  Flexibility: arbitrary python code can be executed inside any sweep
-  Tests:  test coverage  > 80% for all the code in the repository
-  One test suite runs automatically when code is committed, and can
   also be run locally, using simulated instruments
-  Another test suite runs on real instruments, and can be run
   automatically on the full set of instruments loaded for an experiment
-  Scripting API: arbitrary sweeps and collections of sweeps can be
   executed programmatically
-  Standardized data and meta data saving with  gnuplot format and json
-  Realtime visualization
-  Documentation

Phase 2
-------
Broader adoption within stationQ

-  Open source codebase
-  Standardized logging setup, including:
-  raw data (as raw as feasible)
-  analyzed data
-  metadata, including both:
   -  experiment data: all electronic settings and any available manual
      settings, timestamp
   -  information on the code state: git commit id and/or version of
      this package, traceback at the point of execution, and potentially
      information about the other scripts involved in the sweep?
-  automatic logging of monitor parameters (eg fridge and magnet
   information) as a function of time
-  Instrument drivers: all drivers group members have used in their
   existing experiments are ported to the new framework
-  Use semantic versioning
-  pip installable, with simple instructions within a standard python
   installation (likely anaconda)
-  Tests:  100% test coverage

Phase 3
-------
Additional features to help basic users and consolidate the workflow

-  GUI - with a simple control and monitoring panel
-  GUI -> CLI framework, ala Igor, so that each action in the GUI
   generates an equivalent command in the notebook for both record
   keeping and to help users learn the syntax
-  Lab notebook: Additional features as necessary to make the iPython
   Notebook work as the experiment's primary lab notebook.
-  Multiprocessing / distributed control architecture
-  Automated crash/tracebacks reporting to Slack

Phase 4
-------

-  Further polishing as necessary, for example for installation in other
   environments than Anaconda, or to make features more flexible to
   different fields
-  World domination
