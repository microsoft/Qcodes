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

    var SubprocessView = UpdateView.extend({
        render: function() {
            var me = window.SPVIEW = this;
            me._interval = 0;
            me._minimize = '<i class="fa-minus fa"></i>';
            me._restore = '<i class="fa-plus fa"></i>';

            // in case there is already an outputView present,
            // like from before restarting the kernel
            $('.qcodes-output-view').not(me.$el).remove();

            me.$el
                .appendTo('body')
                .addClass('qcodes-output-view')
                .html(
                    '<div class="qcodes-output-header toolbar">' +
                        '<span></span>' +
                        '<button class="btn qcodes-abort-loop disabled">Abort</button>' +
                        '<button class="btn qcodes-clear-output disabled">Clear</button>' +
                        '<button class="btn qcodes-minimize">' + me._minimize + '</button>' +
                    '</div>' +
                    '<pre></pre>'
                );

            me.clearButton = me.$el.find('.qcodes-clear-output');
            me.minButton = me.$el.find('.qcodes-minimize');
            me.outputArea = me.$el.find('pre');
            me.subprocessList = me.$el.find('span');
            me.abortButton = me.$el.find('.qcodes-abort-loop');

            me.clearButton.click(function() {
                me.outputArea.html('');
                me.clearButton.addClass('disabled');
            });

            me.minButton.click(function() {
                if(me.minButton.html() === me._restore) me.restore();
                else me.minimize();
            });

            me.abortButton.click(function() {
                me.send({abort: true});
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

            var processes = this.model.get('_processes') || 'No subprocesses';
            this.abortButton.toggleClass('disabled', processes.indexOf('Measurement')===-1);
            this.subprocessList.text(processes);
        }
    });
    manager.WidgetManager.register_widget_view('SubprocessView', SubprocessView);
});
