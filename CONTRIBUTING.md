# Contributing to Qcodes

## I have an idea!

Great! We all want to know about it. There are different places for different kinds of ideas.

### Bugs reports and feature requests

This is what github's [issues](https://github.com/qdev-dk/Qcodes/issues) are for. Search for existing and closed issues. If your problem or idea is not yet addressed, [please open a new issue](https://github.com/qdev-dk/Qcodes/issues/new)

Choose a label for your issue - please try to use an existing one rather than making a new label. If it involves new functionality, it's an `enhancement`. If it should work but it doesn't, it's a `bug`. Bug reports must be accompanied by a reproducible example.

### It's more than a feature...

Have an idea about future directions to go with Qcodes? Visions of data-utopia that would take more than a few weeks to add or might change some core ideas in the package? We can use issues for this too. Pick the `long-term` or `discussion` labels.

### Clever usage

Figured out a new way to use qcodes? Found a package that makes your life better and easier? Got realtime analysis working after struggling with it for days? Make a new example notebook or script in [docs/examples](https://github.com/qdev-dk/Qcodes/tree/master/docs/examples) and make a [pull request](#pull-requests)

## Development

### Setup

- Clone and register the package for development as described in [README.md#installation]

### Running Tests

The core test runner is in `qcodes/test.py:
```
python qcodes/test.py
# optional extra verbosity
python qcodes/test.py -v
```
You should see output that looks something like this:
```
.........***** found one MockMock, testing *****
............................................Timing resolution:
startup time: 0.000e+00
min/med/avg/max dev: 9.260e-07, 9.670e-07, 1.158e-06, 2.109e-03
async sleep delays:
startup time: 2.069e-04
min/med/avg/max dev: 3.372e-04, 6.376e-04, 6.337e-04, 1.007e-03
multiprocessing startup delay and regular sleep delays:
startup time: 1.636e-02
min/med/avg/max dev: 3.063e-05, 2.300e-04, 2.232e-04, 1.743e-03
should go to stdout;should go to stderr;.stdout stderr stdout stderr ..[10:44:09.063 A Queue] should get printed
...................................
----------------------------------------------------------------------
Ran 91 tests in 4.192s

OK
Name                         Stmts   Miss  Cover   Missing
----------------------------------------------------------
data/data_array.py             104      0   100%
data/data_set.py               179    140    22%   38-55, 79-94, 99-104, 123-135, 186-212, 215-221, 224-244, 251-254, 257-264, 272, 280-285, 300-333, 347-353, 360-384, 395-399, 405-407, 414-420, 426-427, 430, 433-438
data/format.py                 225    190    16%   44-55, 61-62, 70, 78-97, 100, 114-148, 157-188, 232, 238, 246, 258-349, 352, 355-358, 361-368, 375-424, 427-441, 444, 447-451
data/io.py                      76     50    34%   71-84, 90-91, 94, 97, 103, 109-110, 119-148, 154-161, 166, 169, 172, 175-179, 182, 185-186
data/manager.py                124     89    28%   15-20, 31, 34, 48-62, 65-67, 70, 76-77, 80-84, 90-102, 108-110, 117-121, 142-151, 154-182, 185, 188, 207-208, 215-221, 227-229, 237, 243, 249
instrument/base.py              74      0   100%
instrument/function.py          45      1    98%   77
instrument/ip.py                20     12    40%   10-16, 19-20, 24-25, 29-38
instrument/mock.py              63      0   100%
instrument/parameter.py        200      2    99%   467, 470
instrument/sweep_values.py     107     33    69%   196-207, 220-227, 238-252, 255-277
instrument/visa.py              36     24    33%   10-25, 28-32, 35-36, 40-41, 47-48, 57-58, 62-64, 68
loops.py                       285    239    16%   65-74, 81-91, 120-122, 133-141, 153-165, 172-173, 188-207, 216-240, 243-313, 316-321, 324-350, 354-362, 371-375, 378-381, 414-454, 457-474, 477-484, 487-491, 510-534, 537-543, 559-561, 564, 577, 580, 590-608, 611-618, 627-628, 631
station.py                      35     24    31%   17-32, 35, 45-50, 60, 67-82, 88
utils/helpers.py                95      0   100%
utils/metadata.py               13      0   100%
utils/multiprocessing.py        95      2    98%   125, 134
utils/sync_async.py            114      8    93%   166, 171-173, 176, 180, 184, 189-191
utils/timing.py                 72      0   100%
utils/validators.py            110      0   100%
----------------------------------------------------------
TOTAL                         2072    814    61%
```
The key is `OK` in the middle (that means all the tests passed), and the presence of the coverage report after it. If any tests fail, we do not show a coverage report, and the end of the output will contain tracebacks and messages about what failed, for example:
```
======================================================================
FAIL: test_sweep_steps_edge_case (tests.test_instrument.TestParameters)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/alex/qdev/Qcodes/qcodes/tests/test_instrument.py", line 360, in test_sweep_steps_edge_case
    self.check_set_amplitude2('Off', log_count=1, history_count=2)
  File "/Users/alex/qdev/Qcodes/qcodes/tests/test_instrument.py", line 345, in check_set_amplitude2
    self.assertTrue(line.startswith('negative delay'), line)
AssertionError: False is not true : cannot sweep amplitude2 from 0.1 to Off - jumping.

----------------------------------------------------------------------
Ran 91 tests in 4.177s

FAILED (failures=1)
```

The coverage report is only useful if you have been adding new code, to see whether your tests visit all of your code. Look at the file(s) you have been working on, and ensure that the "missing" section does not contain the line numbers of any of the blocks you have touched. Currently the core still has a good deal of untested code - eventually we will have all of this tested, but for now you can ignore all the rest of the missing coverage.

You can also run these tests from inside python. The output is similar except that a) you don't get coverage reporting, and b) one test has to be skipped because it does not apply within a notebook, so the output will end `OK (skipped=1)`:
```python
import qcodes
qcodes.test_core()  # optional verbosity = 1 (default) or 2
```
If the tests pass, you should be ready to start developing!

To tests actual instruments, first instantiate them in an interactive python session, then run `qcodes.test_instruments()`:
```python
import qcodes
sig_gen = qcodes.instrument_drivers.agilent.E8527D.Agilent_E8527D('source', address='...')
qcodes.test_instruments()  # optional verbosity = 1 (default) or 2
```
The output of this command should include lines like:
```
***** found one Agilent_E8527D, testing *****
```
for each class of instrument you have defined. Note that if you instantiate several instruments of the same class, only the *last* one will be tested unless you write the test to explicitly test more than this.

Coverage testing is generally meaningless for instrument drivers, as calls to `add_parameter` and `add_function` do not add any code other than the call itself, which is covered immediately on instantiation rather than on calling these parameters/functions. So it is up to the driver author to ensure that all functionality the instrument supports is covered by tests. Also, it's mentioned below but bears repeating: if you fix a bug, write a test that would have failed before your fix, so we can be sure the bug does not reappear later!

### New code and testing

- Make a branch within this repo, rather than making your own fork. That will be easier to collaborate on.

- Write your new feature or fix. Be sure it doesn't break any existing tests, and please write tests that cover your feature as well, or if you are fixing a bug, write a test that would have failed before your fix. Our goal is 100% test coverage, and although we are not there, we should always strive to increase our coverage with each new feature. Please be aware also that 100% test coverage does NOT necessarily mean 100% logic coverage. If (as is often the case in Python) a single line of code can behave differently for different inputs, coverage in itself will not ensure that this is tested.

- The standard test commands are listed above under [Running Tests](#running_tests). More notes on different test runners can be found in [TESTING.md].

- Core tests live in [qcodes/tests](https://github.com/qdev-dk/Qcodes/tree/master/qcodes/tests) and instrument tests live in the same directories as the instrument drivers.

- We should have a *few* high-level "integration" tests, but simple unit tests (that just depend on code in one module) are more valuable for several reasons:
  - If complex tests fail it's more difficult to tell why
  - When features change it is likely that more tests will need to change
  - Unit tests can cover many scenarios much faster than integration tests.

- If you're having difficulty making unit tests, first consider whether your code could be restructured to make it less dependent on other modules. Often, however, extra techniques are needed to break down a complex test into simpler ones. @alexcjohnson is happy to help with this. Two ideas that are useful here:
  - Patching, one of the most useful parts of the [unittest.mock](https://docs.python.org/3/library/unittest.mock.html) library. This lets you specify exactly how other functions/objects should behave when they're called by the code you are testing. For a simple example, see [test_multiprocessing.py](https://github.com/qdev-dk/Qcodes/blob/58a8692bed55272f4c5865d6ec37f846154ead16/qcodes/tests/test_multiprocessing.py#L63-L65)
  - Supporting files / data: Lets say you have a test of data acquisition and analysis. You can break that up into an acquisition test and an analysis by saving the intermediate state, namely the data file, in the test directory. Use it to compare to the output of the acquisition test, and as the input for the analysis test.

- We have not yet settled on a framework for testing real hardware. Stay tuned, or post any ideas you have as issues!

### Coding Style

- Try to make your code self-documenting. Python is generally quite amenable to that, but some things that can help are:

  - Use clearly-named variables
  - Only use "one-liners" like list comprehensions if they really fit on one line.
  - Comments should be for describing *why* you are doing something. If you feel you need a comment to explain *what* you are doing, the code could probably be rewritten more clearly.
  - If you *do* need a multiline statement, use implicit continuation (inside parentheses or brackets) and implicit string literal concatenation rather than backslash continuation

- Write docstrings for *at least* every public (no leading underscore) method or function. Because docstrings (and comments) *are not code*, pay special attention to them when modifying code: an incorrect comment or docstring is worse than none at all!

- Use [PEP8](http://legacy.python.org/dev/peps/pep-0008/) style. Not only is this style good for readability in an absolute sense, but consistent styling helps us all read each other's code.
  - There is a command-line tool (`pip install pep8`) you can run after writing code to validate its style.
  - A lot of editors have plugins that will check this for you automatically as you type. Sublime Text for example has sublimelinter-pep8 and the even more powerful sublimelinter-flake8.
  - BUT: do not change someone else's code to make it pep8-compliant unless that code is fully tested.

- JavaScript: The [Airbnb style guide](https://github.com/airbnb/javascript) is quite good. If we start writing a lot more JavaScript we can go into more detail.

### Pull requests

- Push your branch back to github and make a pull request (PR). If you visit the repo [home page](https://github.com/qdev-dk/Qcodes) soon after pushing to a branch, github will automatically ask you if you want to make a PR and help you with it.

- Try to keep PRs small and focused on a single task. Frequent small PRs are much easier to review, and easier for others to work around, than large ones that touch the whole code base.

- tag AT LEAST ONE person in the description of the PR (a tag is `@username`) who you would like to have look at your work. Of course everyone is welcome and encouraged to chime in.

- It's OK (in fact encouraged) to open a pull request when you still have some work to do. Just make a checklist (`- [ ] take over the world`) to let others know what more to expect in the near future.

- There are a number of emoji that have specific meanings within our github conversations. The most important one is :dancer: which means "approved" - typically one of the core contributors should give the dancer. Ideally this person was also tagged when you opened the PR.

- You, the initiator of the pull request, should do the actual merge into master after receiving the :dancer: because you will know best if there is anything left you want to add.

- Delete your branch once you have merged (using the helpful button provided by github after the merge) to keep the repository clean. Then on your own computer, after you merge and pull the merged master down, you can call `git branch --merged` to list branches that can be safely deleted, then `git branch -d <branch-name>` to delete it.
