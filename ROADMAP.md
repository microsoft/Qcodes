# Qcodes project plan

The goal is a common framework for all of our experiments, so:
- new students don't need to spend a long time learning software in order to participate in experiments
- nobody has to write their own code except for pieces that are very specific to their own experiment - and even then we will try to merge a lot of this code back into the main project.
- we streamline the process of moving between teams or labs, and of setting up a new experiment.
- we can take advantage of modern software and best practices so the system is easy to maintain and extend as needs and technologies change

We have agreed on python 3 as the foundation of the project - this along with the patchwork of previous efforts means that this will be a breaking change for everyone who migrates to this framework - ie although it will share a good deal of ideas with qtlab, even existing qtlab users will need to change their scripts and workflows to use the new framework.

We expect the project to roll out in roughly four phases:

1. Initial release to tech-savvy experimenters (ie the people who were at the meeting in May) - aiming for 3-4 months in, this will probably involve Alex visiting each team that wants to install it, and working through any issues they have in person.

2. Broader release to anyone within the consortium - 6-8 months

3. Additional features to help basic users and consolidate the workflow - 9-12 months

4. Possible public release as an open-source project - whether, when, and to what extent we do this is still up for debate, and will also depend on how the project itself proceeds.

## Phases

### Phase 1
- Python 3 framework
- Optimized for iPython Notebook, but compatible with terminal shell or standalone scripts run from the command line
- Developed in this github repository (qdev-dk/Qcodes) and all code changes are reviewed
- Modular architecture:
  - Instrument drivers can be easily plugged in
  - Components can be used or not used per each user's preferences
  - Other functions such as visualization packages, data storage back ends, and note-taking capabilities are also easy to swap in or out.
  - Core package, configuration, and experiment scripts are all separate
- Flexibility: arbitrary python code can be executed inside any sweep
- Tests - all code in the repository has tests
  - One test suite runs automatically when code is committed, and can also be run locally, using simulated instruments
  - Another test suite runs on real instruments, and can be run automatically on the full set of instruments loaded for an experiment
- Asynchronous / concurrent operations are natively supported
- Instrument drivers: all drivers group members have used in their existing experiments are ported to the new framework
- Scripting API: arbitrary sweeps and collections of sweeps can be executed programmatically
- Standardized logging setup, including:
  - raw data (as raw as feasible)
  - analyzed data
  - metadata, including both:
    - experiment data: all electronic settings and any available manual settings, timestamp
    - information on the code state: git commit id and/or version of this package, traceback at the point of execution, and potentially information about the other scripts involved in the sweep?
  - automatic logging of monitor parameters (eg fridge and magnet information) as a function of time

### Phase 2
- pip installable, with simple instructions within a standard python installation (likely anaconda)
- Realtime visualization
- Tools to facilitate transfer from other setups people are using
- Documentation: coordinate with Anton
- Project name! **Qcodes** is the working name, courtesy of Guen and Charlie. Any objections? codes == COpenhagen, DElft, and Sydney. I notice that qcodes.com has an expired registration and could probably be picked up if we want it.

### Phase 3
- GUI - with a simple control and monitoring panel
- GUI -> CLI framework, ala Igor, so that each action in the GUI generates an equivalent command in the notebook for both record keeping and to help users learn the syntax
- Lab notebook: Additional features as necessary to make the iPython Notebook work as the experiment's primary lab notebook.
- Multiprocessing / distributed control architecture

### Phase 4
- Decide on the extent to which we will release the code publicly
- Clear contributor guidelines - all contributed code must still be fully tested and reviewed by Alex or other core contributors
- Clear license
- Further polishing as necessary, for example for installation in other environments than Anaconda, or to make features more flexible to different fields
