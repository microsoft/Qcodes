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
            this.display(this.model.get('_message'));
            this.setInterval();
        },
        display: function(message) {
            /*
             * display method: override this for custom display logic
             */
            this.el.innerHTML = message;
        },
        remove: function() {
            clearInterval(this._updater);
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
                    if(!me.model.comm_live) {
                        console.log('missing comm, canceling widget updates', me);
                        clearInterval(me._updater);
                    }
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

    var SubprocessOutputView = UpdateView.extend({
        render: function() {
            var me = window.SPVIEW = this;
            me._interval = 0;
            me._minimize = '&#8212;';
            me._restore = '+';

            // in case there is already an outputView present,
            // like from before restarting the kernel
            $('.qcodes-output-view').not(me.$el).remove();

            me.$el
                .appendTo('body')
                .addClass('qcodes-output-view')
                .css({
                    zIndex: 999,
                    position: 'fixed',
                    bottom: 0,
                    right: '5px'
                }).html(
                    '<div class="qcodes-output-header">' +
                        '<span>Subprocess messages</span>' +
                        '<button class="qcodes-clear-output disabled">Clear</button>' +
                        '<button class="qcodes-minimize">' + me._minimize + '</button>' +
                    '</div>' +
                    '<pre></pre>'
                );

            me.clearButton = me.$el.find('.qcodes-clear-output'),
            me.minButton = me.$el.find('.qcodes-minimize');
            me.outputArea = me.$el.find('pre');

            me.clearButton.click(function() {
                me.outputArea.html('');
                me.clearButton.addClass('disabled');
            });

            me.minButton.click(function() {
                if(me.minButton.html() === me._restore) me.restore();
                else me.minimize();
            });

            me.update();
        },

        minimize: function() {
            this.outputArea.hide();
            this.clearButton.hide()
            this.minButton.html(this._restore);
        },

        restore: function() {
            this.outputArea.show();
            this.clearButton.show();
            this.minButton.html(this._minimize);
        },

        display: function(message) {
            if(message) {
                var initialScroll = this.outputArea.scrollTop();
                this.outputArea.scrollTop(this.outputArea.prop('scrollHeight'));
                var scrollBottom = this.outputArea.scrollTop();

                if(this.minButton.html() === this._restore) {
                    this.restore();
                    // always scroll to the bottom if we're restoring
                    // because there's a new message
                    initialScroll = scrollBottom;
                }

                this.outputArea.append(message);
                this.clearButton.removeClass('disabled');

                // if we were scrolled to the bottom initially, make sure
                // we stay that way.
                this.outputArea.scrollTop(initialScroll === scrollBottom ?
                    this.outputArea.prop('scrollHeight') : initialScroll);
            }
        }
    });
    manager.WidgetManager.register_widget_view('SubprocessOutputView', SubprocessOutputView);
});
