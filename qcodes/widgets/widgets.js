/*
 * Qcodes Jupyter/IPython widgets
 */
require([
    'nbextensions/widgets/widgets/js/widget',
    'nbextensions/widgets/widgets/js/manager'
], function (widget, manager) {

    var UpdateView = widget.DOMWidgetView.extend({
        render: function() {
            window.MYWIDGET = this;
            this._interval = 0;
            this.update();
        },
        update: function() {
            var me = this;

            me.display(me.model.get('_message'));
            me.setInterval();
        },
        display: function(message) {
            /*
             * display method: override this for custom display logic
             */
            this.el.innerHTML = message;
        },
        remove: function() {
            this.setInterval(0);
        },
        setInterval: function(newInterval) {
            var me = this;
            if(newInterval===undefined) newInterval = me.model.get('interval');
            if(newInterval===me._interval) return;

            me._interval = newInterval;

            if(me._updater) clearInterval(me._updater);

            if(me._interval) {
                me._updater = setInterval(function() {
                    me.send({myupdate: true});
                }, me._interval * 1000);
            }
        }
    });
    manager.WidgetManager.register_widget_view('UpdateView', UpdateView);

    var HiddenUpdateView = UpdateView.extend({
        display: function(message) {
            this.$el.hide();
        }
    });
    manager.WidgetManager.register_widget_view('HiddenUpdateView', HiddenUpdateView);
});
