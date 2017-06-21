/*
* init.sql
* Copyright (C) 2017 unga <giulioungaretti@me.com>
*
* Distributed under terms of the MIT license.
* 
* creates the tables that store all the infromations about
* a (set) of experiment(s).
* The idea is to leave the freedom to group experiments
* to a logical "experiment" even if they consists of multiple
* samples.
* TODO(giulioungaretti): 
* - figure out how to store derived dataSets
*/

CREATE TABLE experiments (
    -- this will autoncrement by default if 
    -- no value is specified on insert 
    exp_id INTEGER PRIMARY KEY,
    name TEXT,
    sample_name TEXT,
    start_time INTEGER,
    end_time INTEGER,
    -- this is the last counter registered
    -- 1 based
    run_counter INTEGER,
    -- this is the formatter strin used to cosntruct
    -- the run name
    format_string TEXT
-- TODO: maybe I had a good reason for this doulbe primary key
--    PRIMARY KEY (exp_id, start_time, sample_name)
);

CREATE TABLE runs (
    -- this will autoncrement by default if 
    -- no value is specified on insert 
    run_id INTEGER PRIMARY KEY,
    exp_id INTEGER,
    -- friendly name for the run 
    name TEXT,
    -- the name of the table which stores 
    -- the actual results
    result_table_name TEXT,
    -- this is the run counter in its experiment 0 based
    result_counter INTEGER,
    --- 
    run_timestamp INTEGER,
    parameters TEXT,
    -- metadata fields are added dynamically
    FOREIGN KEY(exp_id) 
    REFERENCES 
        experiments(exp_id)
);

-- Template for run
-- CREATE TABLE majoarana (
--     measurement_id INTEGER,
--     measurement_timestamp INTEGER,
--     x INTEGER,
--     y INTEGER,
--     FOREIGN KEY(measurement_id, measurement_timestamp) 
--     REFERENCES 
--         dataSets(measurement_id, measurement_timestamp)
-- );
