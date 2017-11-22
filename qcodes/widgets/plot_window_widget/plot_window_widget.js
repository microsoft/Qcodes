
require.undef('plot_window');

define('plot_window', ["@jupyter-widgets/base", "notebook/js/codecell"], function(widgets, CodeCell) {

  var CodeCell = CodeCell.CodeCell;

  var PlotWindowView = widgets.DOMWidgetView.extend({

    render: function () {
      console.log('rendering')
      this.model.on('change:_execute_code', this.execute_code, this);
      this.model.on('change:_execute_cell', this.execute_cell, this);
      this.model.on('change:collapsed', this.traitlet_collapsed_event, this);

      var nb = Jupyter.notebook;
      this.notebook = nb;
      this.kernel = nb.kernel;
      this.km = nb.keyboard_manager;
      this.collapsed = true;

      // remove any previous plot window view
      $('[id=nbextension-plotwindowview]').remove()

      // create elements
      this.element = $("<div id='nbextension-plotwindowview'>");
      this.close_button = $("<i>").addClass("fa fa-caret-square-o-down plotwindowview-btn plotwindowview-close");
      this.open_button = $("<i>").addClass("fa fa-caret-square-o-up plotwindowview-btn plotwindowview-open");
      this.element.append(this.close_button);
      this.element.append(this.open_button);
      this.open_button.click(() => {this.expand()});
      this.close_button.click(() => {this.collapse()});

      // create my cell
      var cell = this.cell = new CodeCell(nb.kernel, {
        events: nb.events,
        config: nb.config,
        keyboard_manager: nb.keyboard_manager,
        notebook: nb,
        tooltip: nb.tooltip
      });
      cell._handle_execute_reply = this._handle_execute_reply;

      cell.set_input_prompt();
      this.element.append($("<div/>").addClass('cell-wrapper').append(this.cell.element));
      cell.render();
      cell.refresh();
      this.collapse();

      // override ctrl/shift-enter to execute me if I'm focused instead of the notebook's cell
      var execute_and_select_action = this.km.actions.register({
        handler: $.proxy(this.execute_and_select_event, this)
      }, 'plotwindowview-execute-and-select');
      var execute_action = this.km.actions.register({
        handler: $.proxy(this.execute_event, this)
      }, 'plotwindowview-execute');
      var toggle_action = this.km.actions.register({
        handler: $.proxy(this.toggle, this)
      }, 'plotwindowview-toggle');
      var execute_and_select_action = this.km.actions.register({
        handler: $.proxy(this.execute_and_select_event, this)
      }, 'plotwindowview-execute-and-select');

      var shortcuts = {
        'shift-enter': execute_and_select_action,
        'ctrl-enter': execute_action,
        'ctrl-b': toggle_action
      };
      this.km.edit_shortcuts.add_shortcuts(shortcuts);
      this.km.command_shortcuts.add_shortcuts(shortcuts);

      // finally, add me to the page
      $("body").append(this.element);
      console.log('Rendered plot window view')
    },

    execute_code: function () {
      var cell_text = this.model.get('cell_text');
      this.cell.set_text(cell_text);
      this.execute_cell()
    },

    execute_cell: function () {
      console.log('executing cell');
      this.cell.execute()

    },

    get_code: function () {
      console.log(this.cell.get_text());
      return this.cell.get_text()
    },

    _handle_execute_reply: function (msg) {
      var output_prompt = this.element.find('.out_prompt_overlay.prompt');
      console.log(output_prompt);
      output_prompt.remove();

      var output_prompt = this.element.find('.prompt.output_prompt');
      console.log(output_prompt);
      output_prompt.remove()


      var output_area = this.element.find('.output_area')
      console.log(output_area)
      var output_prompt = output_area.find('.prompt')
      console.log(output_prompt)
      output_prompt.remove()

      var plot = this.element.find('.output_subarea.output_html.rendered_html')
      console.log(plot)
      plot.css('max-width', '100%')
    },

    toggle: function () {
      if (this.collapsed) {
        this.expand();
      } else {
        this.collapse();
      }
      return false;
    },

    expand: function () {
      this.collapsed = false;
      console.log('expanding')
      this.model.set('collapsed', false);
      this.model.save_changes();
      var site_height = $("#site").height();
      this.element.animate({
        height: "auto"
      }, 200);
      this.element.css("min-height", "200px");
      this.element.css("max-height", site_height);
      this.element.css("height", "auto");
      this.open_button.hide();
      this.close_button.show();
      this.cell.element.show();
      this.cell.focus_editor();
      //$("#notebook-container").css('margin-left', 0);


      var input_prompt = this.element.find('.prompt.input_prompt');
      // console.log(input_prompt)
      input_prompt.remove()

    },

    collapse: function () {
      this.collapsed = true;
      console.log('collapsing')
      this.model.set('collapsed', true);
      this.model.save_changes();

      //$("#notebook-container").css('margin-left', 'auto');
      this.element.css("min-height", "");
      this.element.animate({
        height: 0,
      }, 100);
      this.close_button.hide();
      this.open_button.show();
      this.cell.element.hide();
    },

    traitlet_collapsed_event: function () {
      if (this.collapsed != this.model.get('collapsed')){
        if (this.model.get('collapsed')) {
          this.collapse()
        } else {
          this.expand()
        }
      }
    },

    execute_and_select_event: function (evt) {
      if (utils.is_focused(this.element)) {
        this.cell.execute();
      } else {
        this.notebook.execute_cell_and_select_below();
      }
    },

    execute_event: function (evt) {
      if (utils.is_focused(this.element)) {
        this.cell.execute();
      } else {
        this.notebook.execute_selected_cells();
      }
    }
  });

  return {
    PlotWindowView : PlotWindowView
  };


});