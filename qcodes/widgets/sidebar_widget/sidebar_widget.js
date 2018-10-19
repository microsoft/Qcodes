/**
 * Created by Serwan on 20-Sep-17.
 */

require.undef('sidebar');

define('sidebar', ["@jupyter-widgets/base", "notebook/js/codecell"], function(widgets, CodeCell) {
  var CodeCell = CodeCell.CodeCell;

  function log(msg) {
    if (typeof(msg) === 'string') {
      console.log(`[Sidebar] ${msg}`)
    } else {
      console.log(msg)
    }
  }

  var SidebarView = widgets.DOMWidgetView.extend({
    widgetCells:  {},
    position: 'left',

    // Render the view.
    render: function() {

      this.widgetCells = {};
      this.name = this.model.get('name');

      this.model.on('change:_initialize', this.initialize_sidebar, this);
      this.model.on('change:_closed', this.close_sidebar, this);


      this.model.on('change:_add_widget', this.addWidget, this);
      this.model.on('change:_remove_widget', this.removeWidget, this);
      this.model.on('change:_clear_all_widgets', this.clearAllWidgets, this);

      // Add periodic hiding of any additional output
      setInterval(function() {
        let cells = $(".widget-cell");
        cells.each(function(elem, cell) {
          let output_areas = $(cell).find('.output_area');
          output_areas.each(function(output_area_idx, output_area) {
            if (output_area_idx > 0) {
              $(output_area).hide()
            }
          })
        })
      }, 3000)
    },

    initialize_sidebar: function() {
      let position = this.position = this.model.get('position');

      // Remove any previous sidebars with same position
      $(`#sidebar-wrapper-${position}`).remove();

      // Create new sidebar wrapper
      let sidebar = this.sidebar = $(`<divu id="sidebar-wrapper-${position}"/>`);
      sidebar.addClass(`sidebar-wrapper ${position}`);
      $("body").append(sidebar);

      sidebar.css('height', $('#site').height());
      sidebar.css('top', $('#header').height());

      sidebar.resizable({
        handles: "all" ,
        autoHide:true,
        resize : function(event,ui){
          log('resizing');
          setNotebookWidth()
        },
      });

      // Add sidebar height resizing
      $([Jupyter.events]).on("resize-header.Page", () => {this.set_sidebar_height()});
      $(window).on('resize', () => {
        this.set_sidebar_height();
        setNotebookWidth()});

      $(window).trigger('resize');

      log(`${position} sidebar initialized`);
      log(sidebar)

    },

    close_sidebar: function() {
      this.sidebar.remove();
      log(`Closed ${this.position} sidebar`)

    },

    set_sidebar_height: function() {
      var headerVisibleHeight = $('#header').is(':visible') ? $('#header').height() : 0;
      this.sidebar.css('top', headerVisibleHeight);
      this.sidebar.css('height', $('#site').height());
    },

    addWidget: function() {
      let widgetName = this.model.get('_widget_name');

      if (widgetName in this.widgetCells) {
        log(`Widget ${widgetName} already exists, please remove it first`);
      } else {
        log('Adding widget: ' + widgetName);

        let notebook = Jupyter.notebook;
        let cell = new CodeCell(notebook.kernel, {
          events: notebook.events,
          config: notebook.config,
          keyboard_manager: notebook.keyboard_manager,
          notebook: notebook,
          tooltip: notebook.tooltip,
        });
        cell._handle_execute_reply = _cell_handle_execute_reply;
        $(cell.element).addClass('widget-cell').attr('padding', 0);

        cell.set_text(
          `from IPython.display import display\n` +
          `_widget = ${this.name}.widgets['${widgetName}']\n` +
          `_widget.sidebar_position = "${this.position}"\n` +
          `_widget.display() if hasattr(_widget, 'display') else display(_widget)`);
        this.sidebar
          .prepend($("<div/>")
            .append(cell.element));
        cell.execute();
        cell.render();
        cell.refresh();

        this.widgetCells[widgetName] = cell;
      }
    },

    removeWidget: function() {
      let widgetName = this.model.get('_widget_name');
      log('Removing widget: ' + widgetName);
      this.widgetCells[widgetName].element.parent('div').remove();
      delete this.widgetCells[widgetName];

    },

    clearAllWidgets: function() {
      log('clearing all widgets');
      for (var key in this.widgetCells) {
        this.widgetCells[key].element.parent('div').remove()
      }
    }
  });


  function setNotebookWidth() {
    var width = 0;

    let notebook_container = $('#notebook-container');
    let sidebar_wrapper_left = $('#sidebar-wrapper-left');
    if (sidebar_wrapper_left.length) {
      notebook_container.css('margin-left', sidebar_wrapper_left.width() + 30);
      width += sidebar_wrapper_left.width() + 30
    } else {
      notebook_container.css('margin-left', 20)
      width += 20;
    }

    let sidebar_wrapper_right = $('#sidebar-wrapper-right');
    if (sidebar_wrapper_right.length) {
      sidebar_wrapper_right.css('left', $('#notebook').width() - sidebar_wrapper_right.width() -parseInt(sidebar_wrapper_right.css('right'), 10));
      notebook_container.css('margin-right', sidebar_wrapper_right.width() + 30);
      width += sidebar_wrapper_right.width() + 30
    } else {
      notebook_container.css('margin-right', 20)
      width += 20;
    }

    notebook_container.css('width', $('#notebook').width() - width);
  }

  function hideCellElements(cell) {
    cell.element.find('.input').hide();
    cell.element.find('div.out_prompt_overlay.prompt').remove();
    cell.element.find('div.prompt').hide();
    cell.element.find('div.output_area').find('div.prompt').remove();
    cell.element.find('div.output_subarea').css('padding', 0)
    cell.element.find('div.output_subarea.jupyter-widgets-view').css('max-width', '100%')
  }

  function sleep(s) {
    return new Promise(resolve => setTimeout(resolve, s * 1000));
  }

  function _cell_handle_execute_reply(msg) {
    this.element.removeClass("running");
    this.events.trigger('set_dirty.Notebook', {value: true});
    setTimeout(() => {hideCellElements(this)}, 500)
  }

  return {
    SidebarView: SidebarView
  };
});