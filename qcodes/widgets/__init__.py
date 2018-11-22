"""Example initialization of widgets:

```
left_sidebar_widget = SidebarWidget('left_sidebar_widget')
loop_manager_widget = left_sidebar_widget.add_widget('loop_manager', LoopManagerWidget())
toc_widget = left_sidebar_widget.add_widget('toc_widget', TOCWidget())

right_sidebar_widget = SidebarWidget('right_sidebar_widget', 'right')
cell_window_widget = right_sidebar_widget.add_widget('cell_window_widget', CellWindowWidget())
cell_macro_widget = right_sidebar_widget.add_widget('cell_macro_widget', CellMacroWidget())
```
"""

from .display import display_auto
from .cell_window_widget import CellWindowWidget
from .sidebar_widget import SidebarWidget
from .toc_widget import TOCWidget
from .cell_macro_widget import CellMacroWidget
from .widgets import *