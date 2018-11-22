
require.undef('cell_macro');

define('cell_macro', [
  "@jupyter-widgets/base",
  ], function(widgets) {

  var cfg = {};
  var sidebar = undefined;
  var cell_window_cell = undefined;

  var ctrl_key_pressed = false;
  var alt_key_pressed = false;

  function log(msg) {
    if (typeof(msg) === 'string') {
      console.log(`[Cell macro] ${msg}`)
    } else {
      console.log(msg)
    }
  }

  var get_all_data_ids = function(serialized_list) {
    let ids = [];
    serialized_list.forEach(function(elem, id) {
      ids.push(elem['id']);
      if (elem['children'] !== undefined) {
        ids.push.apply(ids, get_all_data_ids(elem['children']))
      }
    });
    let unique_ids = ids.filter(function(item, i, ar){ return ar.indexOf(item) === i; });
    return unique_ids
  };


  var CellMacroView = widgets.DOMWidgetView.extend({
    render: function () {
      log('Rendering Cell Macro widget');
      var nb = Jupyter.notebook;
      this.notebook = nb;
      this.kernel = nb.kernel;
      this.km = nb.keyboard_manager;

      let sidebar_position = this.model.get('sidebar_position');
      sidebar = $(`#sidebar-wrapper-${sidebar_position}`);

      // Initialize config
      cfg = this.initialize_cfg();

      // Load nestable list source code
      $('head').append('<script type="text/javascript"' +
        ' src="https://rawgit.com/dbushell/Nestable/master/jquery.nestable.js"></script>');

      // Remove any pre-existing cell macro widget
      $('#cell-macro').remove();

      // Create nestable list
      this.create_nestable_list();

      let attach_to_window_interval = setInterval(() => {
        let cell_window = $('#nbextension-cellwindowview');
        log('Checking if there is a cell window to attach to');
        if (cell_window.length) {
          clearInterval(attach_to_window_interval);
          this.add_macro_to_cell_window();
        }
      }, 2000);

      $(document).on('keyup keydown', function(e){ctrl_key_pressed = e.ctrlKey} );
      $(document).on('keyup keydown', function(e){alt_key_pressed = e.altKey} );

      log('Finished rendering Cell Macro widget');

    },

    initialize_cfg: function() {
      if (IPython.notebook.metadata.cell_macro === undefined) {
        IPython.notebook.metadata.cell_macro = {}
      }
      var cfg = IPython.notebook.metadata.cell_macro;

      if (cfg.macros === undefined) {
        cfg.macros = []
      }

      // Check if cfg.serialized_list needs to be recreated
      if (cfg.serialized_list === undefined) {
        cfg.serialized_list = []
      }
      let unique_ids = get_all_data_ids(cfg.serialized_list);
      log('Unique ids:');
      log(unique_ids);
      if (Math.max(unique_ids) > cfg.macros.length - 1) {
        log(`Serialized list max id ${Math.max(unique_ids)} is longer than 
             macro length ${cfg.macros.length}. Resetting`);
        cfg.serialized_list = []
      }

      // Create simple list from 0 to number of cell macros
      for (var i = 0; i < cfg.macros.length; i++) {
        if (!unique_ids.includes(i)) {
          log(`Could not find id ${i} in serialized list. Appending.`);
          cfg.serialized_list.push({id: i});
        }
      }
      log(`Config:`);
      log(cfg)
      return cfg
    },

    create_nestable_list: function(){

      var cell_macro = $("<div/>")
        .attr("id", "cell-macro")
        .append($("<div/>")
          .addClass("dd")
          .on('change', function() {
            log('Refreshing serialized list');
            cfg.serialized_list = $(this).nestable('serialize');
          })
          .append(this.create_nestable_sublist(cfg.serialized_list)
          )
        );
      sidebar.append(cell_macro);

      // Add nestable functionality with timeout, loading takes a bit
      setTimeout(function(){
            $('.dd').nestable({ /* config options */ });
          }, 2000
      );

    },

    create_nestable_sublist: function(serialized_list) {
      let dd_list = $("<ol/>")
        .addClass("dd-list")
        .attr('id', "dd-main-list");
      for (let idx in serialized_list) {
        if (idx < cfg.macros.length) {
          let list_item = this.create_list_item(serialized_list[idx]);

          if (serialized_list[idx]['children'] !== undefined) {
              list_item.append(this.create_nestable_sublist(serialized_list[idx]['children']))
          }

          dd_list.append(list_item)
        } else {
            console.log(`Serialized list idx ${idx} not found`)
        }
      }
      return dd_list
    },

    create_list_item: function(serialized_list_item){
      let macro_id = serialized_list_item['id'];
      let macro_label = cfg.macros[macro_id]['label'];
      log(`Adding list item id=${macro_id}: ${macro_label}`);

      let self = this;
      let list_item = $("<li>")
          .addClass("dd-item dd3-item")
          .attr("data-id", macro_id)
          .append($("<div/>")
            .addClass('dd-handle dd3-handle')
            .text('Drag')
          ).append(
            $('<div/>')
              .addClass('dd3-content')
              .attr("data-id", macro_id)
              .text(macro_label)
              .on('click', function(evt) {
                if (alt_key_pressed) {
                  log(`Alt key pressed, removing ${this}`);
                  self.remove_macro($(this).attr("data-id"))
                } else {
                  log(`ctrl key pressed (executes code): ${ctrl_key_pressed}`);
                  self.load_code_into_window_cell($(this).attr("data-id"), ctrl_key_pressed);
                  $('#nbextension-cellwindowview').trigger('expand')
                }
            }));
      return list_item
    },

    add_macro_to_cell_window: function(){
      log('Attaching to cell window toolbar');
      let cell_window = $('#nbextension-cellwindowview');
      let cell_window_toolbar = $('#cell-window-toolbar');
      cell_window_cell = $("#cell-window-cell").data("cell");

      // Remove any pre-existing macro attached to the cell window
      $('[id=cell-window-macro]').remove();

      let cell_window_macro = $('<span/>')
        .attr('id', 'cell-window-macro')
        .css('margin-right', '5px');

      // Macro label that can be changed when clicked
      let macro_label = $('<button/>')
        .addClass("link")
        .attr('id', 'macro-label')
        .text('<Macro label>')
        .on('click', function(evt){
        log(`Changing macro label ${macro_label.text()}`);
        var new_label = prompt("Please enter new label", $("#macro-label").text());
        log(new_label)
        if (new_label) {
          $("#macro-label").text(new_label);
        }
      });
      cell_window_macro.append(macro_label);

      // Macro save button
      let save_button = $('<i>')
        .addClass('fa fa-save')
        .addClass("link")
        .on('click', function(evt){
          this.register_new_macro($('#macro-label').text(), cell_window_cell.get_text())
        }.bind(this));
      cell_window_macro.append(save_button);

      cell_window_toolbar.prepend(cell_window_macro);
    },

    register_new_macro: function(label, code){
      let existing_macro = cfg.macros.find(function (el) {
        return el['label'] === label
      });
      if (existing_macro === undefined){
        log(`Creating new macro ${label}`);
        let id = cfg.macros.push({label: label, code: code}) - 1;
        $('#dd-main-list').append(this.create_list_item({id: id}));
        cfg.serialized_list = $('.dd').nestable('serialize');
      } else {
        log(`Updating existing code for ${label}`);
        existing_macro['code'] = code;
      }
    },

    load_code_into_window_cell: function(id, execute){
      log(`Loading code with id ${id} into window cell`);
      let macro = cfg.macros[id];
      $("#macro-label").text(macro['label']);
      cell_window_cell.set_text(macro["code"]);

      if (execute === true) {
        cell_window_cell.execute();
        log('Executing window cell');
      }
    },

    remove_macro: function(id) {
      cfg.macros.splice(id, 1);
      $("#cell-macro").find(`[data-id=${id}]`).remove();

      // Reduce the id of all list items that are higher than id by one
      $("#cell-macro")
        .find('[data-id]')
        .each(function(idx, el) {
          let el_id = $(el).attr('data-id');
          if (el_id > id) {
              $(el).attr('data-id', el_id - 1);
          }
        });

      cfg.serialized_list = $('.dd').nestable('serialize');
      // TODO update serialization
    }
  });

  return {
    CellMacroView : CellMacroView
  };
});