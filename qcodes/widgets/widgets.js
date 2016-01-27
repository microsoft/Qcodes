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
                .addClass('qcodes-output-view')
                .attr('qcodes-state', 'docked')
                .html(
                    '<div class="qcodes-output-header toolbar">' +
                        '<span></span>' +
                        '<button class="btn qcodes-abort-loop disabled">Abort</button>' +
                        '<button class="btn qcodes-clear-output disabled qcodes-content">Clear</button>' +
                        '<button class="btn js-state qcodes-minimized"><i class="fa-minus fa"></i></button>' +
                        '<button class="btn js-state qcodes-docked"><i class="fa-toggle-up fa"></i></button>' +
                        '<button class="btn js-state qcodes-floated"><i class="fa-arrows fa"></i></button>' +
                    '</div>' +
                    '<pre class="qcodes-content"></pre>'
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

            me.abortButton.click(function() {
                me.send({abort: true});
            });

            me.$el.find('.js-state').click(function() {
                var oldState = me.$el.attr('qcodes-state'),
                    state = this.className.substr(this.className.indexOf('qcodes'))
                        .split('-')[1].split(' ')[0];

                // not sure why I can't pop it out of the widgetarea in render, but it seems that
                // some other bit of code resets the parent after render if I do it there.
                // To be safe, just do it on every state click.
                me.$el.appendTo('body');

                if(oldState === 'floated') {
                    me.$el.draggable('destroy').css({left:'', top: ''});
                }

                me.$el.attr('qcodes-state', state);

                if(state === 'floated') {
                    me.$el.draggable().css({
                        left: window.innerWidth - me.$el.width() - 15,
                        top: window.innerHeight - me.$el.height() - 10
                    });
                }
            });

            $(window).resize(function() {
                if(me.$el.attr('qcodes-state') === 'floated') {
                    var position = me.$el.position(),
                        minVis = 20,
                        maxLeft = window.innerWidth - minVis,
                        maxTop = window.innerHeight - minVis;

                    if(position.left > maxLeft) me.$el.css('left', maxLeft);
                    if(position.top > maxTop) me.$el.css('top', maxTop);
                }
            });

            me.update();
        },

        display: function(message) {
            if(message) {
                var initialScroll = this.outputArea.scrollTop();
                this.outputArea.scrollTop(this.outputArea.prop('scrollHeight'));
                var scrollBottom = this.outputArea.scrollTop();

                if(this.$el.attr('qcodes-state') === 'minimized') {
                    this.$el.find('.qcodes-docked').click();
                    // always scroll to the bottom if we're restoring
                    // because of a new message
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
