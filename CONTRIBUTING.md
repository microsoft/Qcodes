# Contributing to Qcodes

## Opening issues

Search for existing and closed issues. If your problem or idea is not yet addressed, [please open a new issue](https://github.com/qdev-dk/Qcodes/issues/new)

Bug reports must be accompanied by a reproducible example.

## Development

### Setup

- Install git: the [command-line toolset](https://git-scm.com/) is the most powerful but the [desktop GUI from github](https://desktop.github.com/) is also quite powerful

- Clone this repo:
```
git clone https://github.com/qdev-dk/Qcodes.git
```

- Install the dependencies, including for testing (see [README.md#Requirements])

- Run the tests. In the root directory of the repository:
```
nosetests --with-coverage --cover-package=qcodes
```
Note: nose has a note on its homepage that it is no longer being actively maintained, so we may want to change this in the near future.

### New code and testing

- Make a branch within this repo, rather than making your own fork. That will be easier to collaborate on.

- Write your new feature or fix. Be sure it doesn't break any existing tests, and please write tests that cover your feature as well, or if you are fixing a bug, write a test that would have failed before your fix. Our goal is 100% test coverage, and although we are not there, we should always strive to increase our coverage with each new feature. Please be aware also that 100% test coverage does NOT necessarily mean 100% logic coverage. If (as is often the case in Python) a single line of code can behave differently for different inputs, coverage in itself will not ensure that this is tested.

- Simple unit tests at a low level are more valuable than complex high-level tests, mainly because if complex tests fail it's difficult to tell why, but also when features change it is likely that more tests will need to change.

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