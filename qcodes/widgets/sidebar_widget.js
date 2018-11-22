/**
 * Created by Serwan on 20-Sep-17.
 */

require.undef('sidebar');

define('sidebar', ["@jupyter-widgets/base"], function(widgets) {

    var SidebarView = widgets.DOMWidgetView.extend({
        widgetCells:  {},

        // Render the view.
        render: function() {
            // this.model.on('change:_add_widget', this.addWidget, this);
        },

        addWidget: function() {
            let widgetName = this.model.get('_widget_name');
            console.log('Adding widget: ' + widgetName);
            let CodeCell = require('notebook/js/codecell').CodeCell;
            let notebook = Jupyter.notebook;
            let cell = this.cell = new CodeCell(notebook.kernel, {
              events: notebook.events,
              config: notebook.config,
              keyboard_manager: notebook.keyboard_manager,
              notebook: notebook,
              tooltip: notebook.tooltip,
            });

            cell.set_text(widgetName);
            $(`#sidebar-wrapper-${`)
                .prepend($("<div/>")
                         .append(cell.element));
            cell.execute();
            cell.render();
            cell.refresh();

            sleep(0.3).then(() => {
                hideCellElements(cell);
                cell.element.find('.output_area').attr('id', 'sidebar_widget');
                setInterval(() => {
                    clearCellAdditionalOutput(this.cell)}, 1000);
            });

            this.widgetCells[widgetName] = this.cell;
        },

        removeWidget: function() {
            let widgetName = this.model.get('_widget_name');
            console.log('Removing widget: ' + widgetName);
            this.cell.element.remove()

        },

        clearAllWidgets: function() {
            console.log('clearing all widgets')
            $('[id=sidebar_widget]').closest('.cell').parent('div').remove()
        }
    });


    function clearCellAdditionalOutput(cell) {
            let output_elem = cell.element.find('.output');
            output_elem.children().not('#sidebar_widget').remove();
        }

    function hideCellElements(cell) {
        cell.element.find('.input').hide();
        cell.element.find('div.out_prompt_overlay.prompt').remove();
        cell.element.find('div.prompt.output_prompt').hide();
        cell.element.find('div.output_area').find('div.prompt').remove();
        cell.element.find('div.output_subarea.jupyter-widgets-view').css('max-width', '100%')
    }

    function sleep(s) {
      return new Promise(resolve => setTimeout(resolve, s * 1000));
    }

    return {
        SidebarView: SidebarView
    };
});