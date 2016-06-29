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
            var me = this;
            me._interval = 0;
            me._minimize = '<i class="fa-minus fa"></i>';
            me._restore = '<i class="fa-plus fa"></i>';

            // max lines of output to show
            me.maxOutputLength = 500;

            // in case there is already an outputView present,
            // like from before restarting the kernel
            $('.qcodes-output-view').not(me.$el).remove();

            me.$el
                .addClass('qcodes-output-view')
                .attr('qcodes-state', 'docked')
                .html(
                    '<div class="qcodes-output-header toolbar">' +
                        '<div class="qcodes-process-list"></div>' +
                        '<button class="btn qcodes-processlines"><i class="fa-list fa"></i></button>' +
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
            me.subprocessList = me.$el.find('.qcodes-process-list');
            me.abortButton = me.$el.find('.qcodes-abort-loop');
            me.processLinesButton = me.$el.find('.qcodes-processlines')

            me.outputLines = [];

            me.clearButton.click(function() {
                me.outputArea.html('');
                me.clearButton.addClass('disabled');
            });

            me.abortButton.click(function() {
                me.send({abort: true});
            });

            me.processLinesButton.click(function() {
                // toggle multiline process list display
                me.subprocessesMultiline = !me.subprocessesMultiline;
                me.showSubprocesses();
            });

            me.$el.find('.js-state').click(function() {
                var state = this.className.substr(this.className.indexOf('qcodes'))
                        .split('-')[1].split(' ')[0];
                me.model.set('_state', state);
            });

            $(window)
                .off('resize.qcodes')
                .on('resize.qcodes', function() {me.clipBounds();});

            me.update();
        },

        updateState: function() {
            var me = this,
                oldState = me.$el.attr('qcodes-state'),
                state = me.model.get('_state');

            if(state === oldState) return;

            setTimeout(function() {
                // not sure why I can't pop it out of the widgetarea in render, but it seems that
                // some other bit of code resets the parent after render if I do it there.
                // To be safe, just do it on every state click.
                me.$el.appendTo('body');

                if(oldState === 'floated') {
                    console.log('here');
                    me.$el.draggable('destroy').css({left:'', top: ''});
                }

                me.$el.attr('qcodes-state', state);

                if(state === 'floated') {
                    me.$el
                        .draggable({stop: function() { me.clipBounds(); }})
                        .css({
                            left: window.innerWidth - me.$el.width() - 15,
                            top: window.innerHeight - me.$el.height() - 10
                        });
                }

                // any previous highlighting is now moot
                me.$el.removeClass('qcodes-highlight');
            }, 0);

        },

        clipBounds: function() {
            var me = this;
            if(me.$el.attr('qcodes-state') === 'floated') {
                var bounds = me.$el[0].getBoundingClientRect(),
                    minVis = 40,
                    maxLeft = window.innerWidth - minVis,
                    minLeft = minVis - bounds.width,
                    maxTop = window.innerHeight - minVis;

                if(bounds.left > maxLeft) me.$el.css('left', maxLeft);
                else if(bounds.left < minLeft) me.$el.css('left', minLeft);

                if(bounds.top > maxTop) me.$el.css('top', maxTop);
                else if(bounds.top < 0) me.$el.css('top', 0);
            }
        },

        display: function(message) {
            var me = this;
            if(message) {
                var initialScroll = me.outputArea.scrollTop();
                me.outputArea.scrollTop(me.outputArea.prop('scrollHeight'));
                var scrollBottom = me.outputArea.scrollTop();

                if(me.$el.attr('qcodes-state') === 'minimized') {
                    // if we add text and the box is minimized, highlight the
                    // title bar to alert the user that there are new messages.
                    // remove then add the class, so we get the animation again
                    // if it's already highlighted
                    me.$el.removeClass('qcodes-highlight');
                    setTimeout(function(){
                        me.$el.addClass('qcodes-highlight');
                    }, 0);
                }

                var newLines = message.split('\n'),
                    out = me.outputLines,
                    outLen = out.length;
                if(outLen) out[outLen - 1] += newLines[0];
                else out.push(newLines[0]);

                for(var i = 1; i < newLines.length; i++) {
                    out.push(newLines[i]);
                }

                if(out.length > me.maxOutputLength) {
                    out.splice(0, out.length - me.maxOutputLength + 1,
                        '<<< Output clipped >>>');
                }

                me.outputArea.text(out.join('\n'));
                me.clearButton.removeClass('disabled');

                // if we were scrolled to the bottom initially, make sure
                // we stay that way.
                me.outputArea.scrollTop(initialScroll === scrollBottom ?
                    me.outputArea.prop('scrollHeight') : initialScroll);
            }

            me.showSubprocesses();
            me.updateState();
        },

        showSubprocesses: function() {
            var me = this,
                replacer = me.subprocessesMultiline ? '<br>' : ', ',
                processes = (me.model.get('_processes') || '')
                    .replace(/\n/g, '&gt;' + replacer + '&lt;');

            if(processes) processes = '&lt;' + processes + '&gt;';
            else processes = 'No subprocesses';

            me.abortButton.toggleClass('disabled', processes.indexOf('Measurement')===-1);

            me.subprocessList.html(processes);
        }
    });
    manager.WidgetManager.register_widget_view('SubprocessView', SubprocessView);
});
