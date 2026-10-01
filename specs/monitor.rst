=====================
Monitor Specification
=====================

Introduction
============

A simple passive monitor (like the name implies) that runs in the background and streams to the outside world
the "value" of a (set) of parameters.


The GUI for the monitor is not part of qcodes.

Monitoring is implemented by reading the last know value of that parameter.

Requirements
============

The Monitor class should meet the following requirements:

Basics
---------

#. Monitor any type and number of parameters
#. Groups by instrument or lack thereof
#. Cheap messaging
#. Non blocking
#. Thread safe

Creation
------------

#. A monitor can be created at any point
#. There can be only one monitor running
#. The websocket address:port can be specified in the config as :

   :: 

     monitor_addres="address:port"


API
---

#. Monitor(*parameters)
   Creates and starts a thread that streams json over a websocket.
   If a monitor is already running, it will be stopped.

#. Monitor.show()
   Raises the window / shows the window.
   This must be implemented by the users.
   

#. JSON endpoint
   The websocket streams the following data structure:

   ::

    Message =
    {
        ts: Float  # when did the monitor last stent data
        parameters: List Parameters
    }

    Parameters =
        { instrument : Maybe String
        , parameters : List Parameter
        }


    Parameter =
        { name : String
        , unit : Maybe String
        , value : Maybe String
        , ts : Maybe Float # when was the parameter value last updated
        }


Open Issues
===========

# Integration with the station
