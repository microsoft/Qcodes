
require.undef('cell_window');

define('cell_window', [
  "@jupyter-widgets/base",
  "notebook/js/codecell",
  ], function(widgets, CodeCell) {

  var CodeCell = CodeCell.CodeCell;

  function log(msg) {
    if (typeof(msg) === 'string') {
      console.log(`[Cell window] ${msg}`)
    } else {
      console.log(msg)
    }
  }

  function sleep(s) {
    return new Promise(resolve => setTimeout(resolve, s * 1000));
  }

  let CellWindowView = widgets.DOMWidgetView.extend({

    render: function () {
      log('rendering');
      this.model.on('change:_execute_code', this.execute_code, this);
      this.model.on('change:_execute_cell', this.execute_cell, this);
      this.model.on('change:collapsed', this.traitlet_collapsed_event, this);

      this.notebook = Jupyter.notebook;
      this.kernel = this.notebook.kernel;
      this.km = this.notebook.keyboard_manager;
      this.collapsed = true;
      this.reference_cell = undefined;

      // remove any previous cell window view
      $('[id=nbextension-cellwindowview]').remove();
      // Remove any previous window cells (appended to bottom of notebook)
      let cells = this.notebook.get_cells();
      for (let k in cells) {
        let cell = cells[k];
        if (cell.metadata.window_cell === true) {
          cell.metadata.deletable = undefined;
          this.notebook.delete_cell(this.notebook.find_cell_index(cell))
        }
      }

      // create main cell window elements
      this.element = this.create_cell_window();

      // Create cell window toolbar
      this.toolbar = this.create_toolbar();
      this.element.append(this.toolbar);

      // Bind to notebook events
      this.bind_events();

      // create main code cell
      this.cell = this.create_cell();
      this.element.append($("<div/>")
        .addClass('cell-wrapper')
        .append(this.cell.element));
      this.cell.render();
      this.cell.refresh();

      // Register keyboard actions
      this.register_keyboard_actions();

      // finally, add me to the page
      this.collapse();
      $("#notebook-container").append(this.element);

      log('Rendered cell window view');
    },

    execute_code: function () {
      // Update reference cell
      this.reference_cell = this.notebook.get_selected_cell();

      let cell_text = this.model.get('cell_text');
      this.cell.set_text(cell_text);
      log('executing code');
      this.execute_cell()
    },

    execute_cell: function () {
      log('executing cell');
      this.cell.execute()

    },

    get_code: function () {
      log(this.cell.get_text());
      return this.cell.get_text()
    },

    create_cell_window: function() {
      let element = $("<div id='nbextension-cellwindowview'>");
      this.close_button = $("<i>").addClass("fa fa-caret-square-o-down cellwindowview-btn cellwindowview-close");
      this.open_button = $("<i>").addClass("fa fa-caret-square-o-up cellwindowview-btn cellwindowview-open");
      element.append(this.close_button);
      element.append(this.open_button);
      this.open_button.click(() => {this.expand()});
      this.close_button.click(() => {this.collapse()});
      element.bind('expand', () => this.expand());
      return element
    },

    create_cell: function() {
      let cell = new CodeCell(this.notebook.kernel, {
        events: this.notebook.events,
        config: this.notebook.config,
        keyboard_manager: this.notebook.keyboard_manager,
        notebook: this.notebook,
        tooltip: this.notebook.tooltip
      });

      this.patch_cell(cell);

      cell.set_input_prompt();
      cell.metadata.deletable = false;
      cell.metadata.window_cell = true;
      cell.element
        .addClass('window-cell')
        .attr('id', 'cell-window-cell');
      return cell
    },

    _patch_cell_handle_execute_reply: function (msg) {
      $(this.element).find('.input_area').css('background-color', '#f7f7f7');
      this.element.find('.out_prompt_overlay.prompt').remove();
      this.element.find('.prompt.output_prompt').remove();
      this.element.find('.output_area').find('.prompt').remove();
      this.element.find('.output_subarea.output_html.rendered_html')
        .css('max-width', '100%')
    },

    bind_events: function() {
      this.notebook.events.on('create.Cell', function (event, data) {
        log(`New cell created at index ${data.index}`);
        if (data.cell.element.parents('#nbextension-cellwindowview').length){
          let selected_cell = this.notebook.get_selected_cell();
          if (!selected_cell.element.parents('#nbextension-cellwindowview').length) {
            log(`Move cell out of cellwindowview`);
            $('#nbextension-cellwindowview').before($(data.cell.element))
          }
        }
      }.bind(this))
    },

    patch_cell: function(cell) {
      // Override cell.execute to also change color to red when busy
      let execute_cell = cell.execute;
      cell.execute = function(stop_on_error){
        $(cell.element).find('.input_area').css('background-color', '#ffcac4');
        execute_cell.apply(cell, stop_on_error)
      };
      // Override cell handle execute reply to stop being red when finished
      cell._handle_execute_reply = this._patch_cell_handle_execute_reply;
    },

    register_keyboard_actions: function() {
      // override ctrl/shift-enter to execute me if I'm focused instead of the notebook's cell
      var toggle_action = this.km.actions.register({
        handler: $.proxy(this.toggle, this)
      }, 'cellwindowview-toggle');
      var execute_and_select_action = this.km.actions.register({
        handler: $.proxy(this.execute_and_select_event, this)
      }, 'cellwindowview-execute-and-select');

      var shortcuts = {
        'shift-enter': execute_and_select_action,
        'ctrl-b': toggle_action
      };
      this.km.edit_shortcuts.add_shortcuts(shortcuts);
      this.km.command_shortcuts.add_shortcuts(shortcuts);

    },

    create_toolbar: function() {
      let toolbar = $('<div/>')
        .attr('id', 'cell-window-toolbar')
        .append($('<i>')
          .addClass('fa fa-eye')
          .addClass("link")
          .on('click', function(evt){
            if (this.reference_cell !== undefined) {
              let index = this.notebook.find_cell_index(this.reference_cell);
              log(`Going to cell index ${index}`);
              this.notebook.select(index, true);
              setTimeout(() => {
                this.notebook.scroll_to_cell(index, 400)
              }, 1000);
            }
          }.bind(this)));
        return toolbar
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
      log('expanding');
      this.model.set('collapsed', false);
      this.model.save_changes();
      var site_height = $("#site").height();
      this.element.animate({
        height: "auto"
      }, 200);
      this.element.css("min-height", "200px");
      this.element.css("max-height", site_height);
      this.element.css("height", "auto").trigger("resize");
      this.open_button.hide();
      this.close_button.show();
      this.cell.element.show();
      this.cell.focus_editor();
      //$("#notebook-container").css('margin-left', 0);


      var input_prompt = this.element.find('.prompt.input_prompt');
      input_prompt.remove()

    },

    collapse: function () {
      this.collapsed = true;
      log('collapsing');
      this.model.set('collapsed', true);
      this.model.save_changes();

      //$("#notebook-container").css('margin-left', 'auto');
      this.element.height(this.element.height());
      this.element.css("min-height", "");
      this.element.animate({
        height: 0,
      }, 200, () => {this.element.trigger("resize")});
      this.close_button.hide();
      this.open_button.show();
      this.cell.element.hide();
    },

    traitlet_collapsed_event: function () {
      if (this.collapsed !== this.model.get('collapsed')){
        if (this.model.get('collapsed')) {
          this.collapse()
        } else {
          this.expand()
        }
      }
    },

    execute_and_select_event: function (evt) {
      if (utils.is_focused(this.cell.element)) {
        this.cell.execute();
      } else {
        this.notebook.execute_cell_and_select_below();
      }
    }

  });

  return {
    CellWindowView : CellWindowView
  };


});