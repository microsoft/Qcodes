/**
 * Created by serwa_000 on 16-Nov-17.
 */
// Adapted from https://gist.github.com/magican/5574556
// by minrk https://github.com/minrk/ipython_extensions
// See the history of contributions in README.md

require.undef('toc');

define('toc', [
  "@jupyter-widgets/base",
  "base/js/events",
  "notebook/js/codecell"
], function(
  widgets,
  events,
  CodeCell
) {
  "use strict";


  var CodeCell = CodeCell.CodeCell;

  function log(msg) {
    if (typeof(msg) === 'string') {
      console.log(`[Sidebar] ${msg}`)
    } else {
      console.log(msg)
    }
  }

  /** ****** **/
  /** Config **/
  /** ****** **/
  var default_cfg = {'threshold':4,
    collapse_to_match_collapsible_headings: true,
    nav_menu: {},
    number_sections: true,
    skip_h1_title: false,
    toc_position: {},
    toc_section_display: 'block',
    toc_window_display: false,
    hide_others: true
  };
  var cfg = {};

  var TOCView = widgets.DOMWidgetView.extend({
    render: function() {

      // Remove any pre-existing TOC
      $('[id=toc-wrapper]').remove();

      if (IPython.notebook.metadata.sidebar_toc === undefined) {
        IPython.notebook.metadata.sidebar_toc = default_cfg;
      }
      cfg = IPython.notebook.metadata.sidebar_toc;

      create_toc_div();
      create_toc_links();

      // Highlight cell on code cell execution
      patch_CodeCell_get_callbacks();

      // Attach TOC functions to IPython and Jupyter events
      attach_events();

      patch_keyboard_actions();

      $('a[toc-id]').each(function() {
        let toc_link_level = get_toc_link_level($(this));
        collapse_by_toc_id($(this).attr('toc-id'))
      });


      if ($("#toc_button").length === 0) {
        IPython.toolbar.add_buttons_group([
          {
            'label'   : 'Show all cells',
            'icon'    : 'fa-eye',
            'callback':  show_all_cells,
            'id'      : 'toc_button'
          }
        ]);
      };

    },


  });


  /** ***************** **/
  /** Cell manipulation **/
  /** ***************** **/
  /** Hide all cells except for certain ids **/
  function hide_all_cells_except_id(toc_id) {
    let toc_num = toc_id.split('-').pop();
    for (let cell of Jupyter.notebook.get_cells()) {
      if ($(cell.element).hasClass('collapsible_headings_ellipsis')) {
        let cell_toc_id = $(cell.element).find('[toc-id]').attr('toc-id');
        if (cell_toc_id !== undefined) {
          let cell_toc_num = cell_toc_id.split('-').pop();

          if (toc_num === cell_toc_num) {
            events.trigger('uncollapse.Toc', {cell: cell});
            cell.element.slideDown('fast')
          } else if (toc_num.startsWith(cell_toc_num + '.')) {
            // is parent
            events.trigger('uncollapse.Toc', {cell: cell});
            cell.element.slideDown('fast')
          } else if (cell_toc_num.startsWith(toc_num)) {
            // is child
            events.trigger('collapse.Toc', {cell: cell});
            cell.element.slideDown('fast')
          } else {
            events.trigger('collapse.Toc', {cell: cell});
            cell.element.slideUp('fast')
          }
        } else {
            events.trigger('collapse.Toc', {cell: cell});
            cell.element.slideUp('fast')
        }

      }
    }
    // var cell_begin = $(document.getElementById(toc_id)).closest('.cell').data('cell');
    // var level = get_cell_level(cell_begin);
    //
    // var cell_end = cell_begin;
    // var next_cell = Jupyter.notebook.get_next_cell(cell_end);
    // while (next_cell !== null && get_cell_level(next_cell) > level) {
    //   cell_end = next_cell;
    //   next_cell = Jupyter.notebook.get_next_cell(cell_end);
    // }
    //
    // hide_cells_above(cell_begin);
    // hide_cells_below(cell_end);
    // show_cells_between(cell_begin, cell_end);
  }

  function hide_cells_above(cell) {
    log(`Hiding cells above ${Jupyter.notebook.find_cell_index(cell)}`);
    while (cell !== null) {
      cell = Jupyter.notebook.get_prev_cell(cell);
      if (cell !== null) {
        cell.element.slideUp(0)
      }
    }
  }

  function hide_cells_below(cell) {
    log(`Hiding cells below ${Jupyter.notebook.find_cell_index(cell)}`);
    while (cell !== null) {
      cell = Jupyter.notebook.get_next_cell(cell);
      if (cell !== null) {
        cell.element.slideUp(0)
      }
    }
  }

  function show_all_cells() {
    log('Showing all cells');
    for (let cell of Jupyter.notebook.get_cells()) {
      cell.element.slideDown(0)
    }
  }

  function show_initialization_cells() {
    log('Showing all initialization cells');
    for (let cell of Jupyter.notebook.get_cells()) {
      if (cell.metadata.init_cell === true) {
        cell.element.slideDown(0)
      } else {
        cell.element.slideUp(0)
      }
    }
  }

  /** Show all cells between two cells, inclusive begin and end **/
  function show_cells_between(cell_begin, cell_end) {
    var cell_begin_index = Jupyter.notebook.find_cell_index(cell_begin);
    var cell_end_index = Jupyter.notebook.find_cell_index(cell_end);
    log(`showing cells between ${cell_begin_index} and ${cell_end_index}`);

    var cell = cell_begin;
    while (Jupyter.notebook.find_cell_index(cell) !== cell_end_index) {
      cell.element.slideDown(0);
      delete cell.metadata.heading_collapsed;
      cell = Jupyter.notebook.get_next_cell(cell);
    }
    cell.element.slideDown(0);
  }

  /**
   * Return the level of nbcell.
   * The cell level is an integer in the range 1-7 inclusive
   *
   * @param {Object} cell Cell instance or jQuery collection of '.cell' elements
   * @return {Integer} cell level
   */
  function get_cell_level (cell) {
    // headings can have a level up to 6, so 7 is used for a non-heading
    var level = 7;
    if (cell === undefined) {
      return level;
    }
    if ((typeof(cell) === 'object')  && (cell.cell_type === 'markdown')) {
      level = cell.get_text().match(/^#*/)[0].length || level;
    }
    return Math.min(level, 7); // we rely on 7 being max
  }

  /** ************ **/
  /** TOC Creation **/
  /** ************ **/
  /** Create toc division in sidebar (header but no links) **/
  function create_toc_div() {
    var toc_wrapper = $('<div id="toc-wrapper"/>')
      .append(
        $('<div id="toc-header"/>')
          .addClass("header")
          .text("Contents ").append(
          $("<a/>")
            .attr("href", "#")
            .addClass("reload-btn")
            .text("  \u21BB")
            .attr('title', 'Reload ToC')
            .click( function(){
              create_toc_links();
              return false;
            })
        ).append(
          $("<span/>")
            .html("&nbsp;&nbsp")
        ).append(
          $("<a/>")
            .attr("href", "#")
            .addClass("number_sections-btn")
            .text("n")
            .attr('title', 'Number text sections')
            .click( function(){
              cfg.number_sections=!(cfg.number_sections);
              log(`number sections: ${cfg.number_sections}`);

              IPython.notebook.set_dirty();
              // $('.toc-item-num').toggle();
              cfg.number_sections ? $('.toc-item-num').show() : $('.toc-item-num').hide();
              //create_toc_links();
              return false;
            })
        ).append(
          $("<span/>")
            .html("&nbsp;&nbsp")
        ).append(
          $("<a/>")
            .attr("href", "#")
            .addClass("hide_others-btn")
            .text("H")
            .attr('title', 'Hide other headers')
            .click( function(){
              cfg.hide_others=!(cfg.hide_others);
              if (!cfg.hide_others) {
                show_all_cells();
              }
            })
        ).append(
          $("<span/>")
            .html("&nbsp;&nbsp")
        ).append(
          $("<a/>")
            .attr("href", "#")
            .addClass("initialize-btn")
            .text("I")
            .attr('title', 'Show initialization cells')
            .click( function(){
                show_initialization_cells();
              }
            )
        )
      ).append(
        $("<div/>").attr("id", "toc").addClass('toc')
      );

    // TODO: either left or right sidebar
    var sidebar = $('#sidebar-wrapper-left').append(toc_wrapper);

    // Restore toc display
    $('#toc').css('display','block');
    // $('#toc').css('height', $('#toc-wrapper').height()-$('#toc-header').height());
  }

  /** Create TOC links **/
  function create_toc_links() {

    var toc_wrapper = $("#toc-wrapper");

    // Create TOC div if not yet created
    if (toc_wrapper.length === 0) {
      create_toc_div();
    }

    var ul = $("<ul/>").addClass("toc-item").attr('id','toc-level0');

    // update toc element
    $("#toc").empty().append(ul);

    var depth = 1; //var depth = ol_depth(ol);
    var li= ul;//yes, initialize li with ul!
    var all_headers= $("#notebook").find(":header");
    var min_lvl = 1 + Number(Boolean(cfg.skip_h1_title)), lbl_ary = [];
    for(; min_lvl <= 6; min_lvl++){ if(all_headers.is('h'+min_lvl)){break;} }
    for(var i= min_lvl; i <= 6; i++){ lbl_ary[i - min_lvl]= 0; }

    //loop over all headers
    all_headers.each(function (i, h) {
      var level = parseInt(h.tagName.slice(1), 10) - min_lvl + 1;
      // skip below threshold, or h1 ruled out by cfg.skip_h1_title
      if (level < 1 || level > cfg.threshold){ return; }
      // skip headings with no ID to link to
      if (!h.id){ return; }
      // skip toc cell if present
      if (h.id=="Table-of-Contents"){ return; }
      // skip header if an html tag with class 'tocSkip' is present
      // eg in ## title <a class='tocSkip'>
      if ($(h).find('.tocSkip').length != 0 ) {
        return; }
      h = $(h);
      h.children('.toc-item-num').remove(); // remove pre-existing number
      // numbered heading labels
      var num_str = incr_lbl(lbl_ary, level - 1).join('.');

      var item_num = $('<span>')
        .text(num_str + '\u00a0\u00a0')
        .addClass('toc-item-num')
        .prependTo(h);

      if (!cfg.number_sections) item_num.hide();

      // walk down levels
      for(var elm=li; depth < level; depth++) {
        var new_ul = $("<ul/>").addClass("toc-item");
        elm.append(new_ul);
        elm= ul= new_ul;
      }
      // walk up levels
      for(; depth > level; depth--) {
        // up twice: the enclosing <ol> and <li> it was inserted in
        ul= ul.parent();
        while(!ul.is('ul')){ ul= ul.parent(); }
      }

      var toc_mod_id = h.attr('id') + '-' + num_str;
      h.attr('toc-id', toc_mod_id);
      // add an anchor with modified id (if it doesn't already exist)
      h.children('.toc-mod-link').remove();
      $('<a>').addClass('toc-mod-link').attr('id', toc_mod_id).prependTo(h);

      // Create lock for hiding of cells
      // var lock = $('<i>')
      //   .addClass('fa fa-unlock')
      //   .css('opacity', 0.1)
      //   .on('click', function(evt){
      //     if ($(evt.currentTarget).hasClass('fa-unlock')) {
      //       $(evt.currentTarget).attr('class', 'fa fa-lock')
      //         .css('opacity', 1)
      //     } else {
      //       $(evt.currentTarget).attr('class', 'fa fa-unlock')
      //         .css('opacity', 0.1)
      //     }
      //   });

      li = $('<li>')
        .append($('<span/>')
          .css('display', 'flex')
          .append(make_link(h, toc_mod_id))
          // .append(lock)
        );
      ul.append(li);
    });

    // add collapse controls
    $('<i>')
      .addClass('fa fa-fw fa-caret-down')
      .on('click', callback_collapser) // callback
      .prependTo('.toc li:has(ul) > span');   // only if li has descendants
    $('<i>').addClass('fa fa-fw ').prependTo('.toc li:not(:has(ul)) > span');    // otherwise still add <i> to keep things aligned
  }

  /** Add TOC link **/
  function make_link(h, toc_mod_id) {
    log('making TOC link (make_link)');
    var a = $('<a>')
      .attr({
        'href': window.location.href.split('#')[0] + h.find('.anchor-link').attr('href'),
        'toc-id': toc_mod_id,
      });
    // get the text *excluding* the link text, whatever it may be
    var hclone = h.clone();
    hclone = remove_MathJax_preview(hclone);
    a.html(hclone.html());
    a.width('100%');
    a.css('display', 'inline-block');
    a.on('click', callback_toc_link_click);
    return a;
  }


  /** **************** **/
  /** TOC manipulation **/
  /** **************** **/
  /** Highlight a TOC element, either as executing or selected **/
  function highlight_toc_item(evt, data) {
    // log('Highlighting toc item (highlight_toc_item)');
    var c = $(data.cell.element);
    if (c.length < 1) {
      return;
    }
    var toc_id = c.find('.toc-mod-link').attr('id') ||
      c.prevAll().find('.toc-mod-link').eq(-1).attr('id');
    var highlighted_item = $();
    if (toc_id !== undefined) {
      highlighted_item = $('.toc a').filter(function (idx, elt) {
        return $(elt).attr('toc-id') === toc_id;
      });
    }
    if (evt.type === 'execute') {
      // remove the selected class and add execute class
      // if the cell is selected again, it will be highligted as selected+running
      highlighted_item.removeClass('toc-item-highlight-select').addClass('toc-item-highlight-execute');
    }
    else {
      $('.toc .toc-item-highlight-select').removeClass('toc-item-highlight-select');
      highlighted_item.addClass('toc-item-highlight-select');
    }
  }

  /** Perform actions when TOC link is clicked **/
  function callback_toc_link_click(evt) {
    // workaround for https://github.com/jupyter/notebook/issues/699
    setTimeout(function () {
      $.ajax()
    }, 100);
    evt.preventDefault();

    let current_toc_link = $(evt.currentTarget);
    let toc_id = current_toc_link.attr('toc-id');
    log('clicked TOC link with id ' + toc_id);

    document.getElementById(toc_id).scrollIntoView(true);



    // Select first cell in link
    var cell = $(document.getElementById(toc_id)).closest('.cell').data('cell');
    Jupyter.notebook.select(Jupyter.notebook.find_cell_index(cell));

    // Hide all cells not in toc_link
    if (cfg.hide_others) {
      hide_all_cells_except_id(toc_id);
    }

    // collapse all other toc links
    $('a[toc-id]').each(function(idx, toc_link) {
      let toc_id = $(toc_link).attr('toc-id')
      if (!toc_link_is_parent(toc_link, current_toc_link) && !(toc_link === current_toc_link)) {
        collapse_by_toc_id(toc_id, false);
        log('collapsing toc link ' + $(toc_link).attr('toc-id'))
      } else {
        collapse_by_toc_id(toc_id, true)
      }
    });

    highlight_toc_item("toc_link_click", {cell: cell});
  }

  function rehighlight_running_cells() {
    $.each($('.running'), // re-highlight running cells
      function(idx, elt) {
        highlight_toc_item({ type: "execute" }, $(elt).data())
      }
    )
  }

  /** increment heading label  w/ h_idx (zero based) **/
  function incr_lbl(ary, h_idx) {
    ary[h_idx]++;
    for (var j = h_idx + 1; j < ary.length; j++) { ary[j] = 0; }
    return ary.slice(0, h_idx + 1);
  }

  /** Remove MathJax for links (no Latex) **/
  function remove_MathJax_preview(elt) {
    elt.children('.anchor-link, .toc-mod-link').remove();
    elt.find("script[type='math/tex']").each(
      function(i, e) {
        $(e).replaceWith('$' + $(e).text() + '$')
      });
    elt.find("span.MathJax_Preview").remove();
    elt.find("span.MathJax").remove();
    return elt
  }

  /** Collapse TOC link **/
  function collapse_by_toc_id(toc_id, show, trigger_event) {
    var anchors = $('.toc .toc-item > li > span > a').filter(function (idx, elt) {
      return $(elt).attr('toc-id') === toc_id;
    });
    anchors.siblings('i')
      .toggleClass('fa-caret-right', !show)
      .toggleClass('fa-caret-down', show);
    anchors.parent().siblings('ul')[show ? 'slideDown' : 'slideUp']('fast');
    if (trigger_event !== false) {
      // fire event for collapsible_heading to catch
      var cell = $(document.getElementById(toc_id)).closest('.cell').data('cell');

    }
  }

  function callback_sidebar_toc_collapsible_headings(evt, data) {
    var toc_id = data.cell.element.find(':header').filter(function (idx, elt) {
      return Boolean($(elt).attr('toc-id'));
    }).attr('toc-id');
    var show = evt.type.indexOf('un') >= 0;
    // use trigger_event false to avoid re-triggering collapsible_headings
    collapse_by_toc_id(toc_id, show, false);
  }

  function callback_collapser(evt) {
    var clicked_i = $(evt.currentTarget);
    var toc_id = clicked_i.siblings('a').attr('toc-id');
    $('.toc .toc-item > li > span > a').filter(function (idx, elt) {
      return $(elt).attr('toc-id') === toc_id;
    });
    var show = clicked_i.hasClass('fa-caret-right');
    collapse_by_toc_id(toc_id, show);
  }

  /** Check if toc link is parent of other toc link **/
  function toc_link_is_parent(current_toc_link, target_toc_link) {
    let current_toc_id = $(current_toc_link).attr('toc-id').split('-').pop();
    let target_toc_id = $(target_toc_link).attr('toc-id').split('-').pop();
    return target_toc_id.startsWith(current_toc_id)
  }

  function get_toc_link_level(toc_link) {
    let toc_id = $(toc_link).attr('toc-id').split('-').pop();
    return toc_id.split('.').length;
  }


  /** *************************** **/
  /** Jupyter events and patching **/
  /** *************************** **/
  function attach_events() {
    // event: render toc for each markdown cell modification
    $([IPython.events]).off("rendered.MarkdownCell");
    $([IPython.events]).on("rendered.MarkdownCell",
      function(evt, data) {
        create_toc_links(); // recompute the toc
        rehighlight_running_cells(); // re-highlight running cells
        highlight_toc_item(evt, data); // and of course the one currently rendered
      });

    // event: on cell selection, highlight the corresponding item
    $([IPython.events]).on('select.Cell', highlight_toc_item);

    $([Jupyter.events]).on('execute.CodeCell', highlight_toc_item);

    // turn off event where delete cell unhides all other cells
    setTimeout(function() {
        Jupyter.notebook.events.off('delete.Cell')
    }, 5000);


    events[cfg.collapse_to_match_collapsible_headings ? 'on' : 'off'](
      'collapse.CollapsibleHeading uncollapse.CollapsibleHeading', callback_sidebar_toc_collapsible_headings);
  }

  function patch_CodeCell_get_callbacks() {
    var previous_get_callbacks = CodeCell.prototype.get_callbacks;
    CodeCell.prototype.get_callbacks = function() {
      var callbacks = previous_get_callbacks.apply(this, arguments);
      var prev_reply_callback = callbacks.shell.reply;
      callbacks.shell.reply = function(msg) {
        if (msg.msg_type === 'execute_reply') {
          setTimeout(function(){
            $('.toc .toc-item-highlight-execute').removeClass('toc-item-highlight-execute');
            rehighlight_running_cells() // re-highlight running cells
          }, 100);
          var c = IPython.notebook.get_selected_cell();
          highlight_toc_item({ type: 'selected' }, { cell: c })
        }
        return prev_reply_callback(msg);
      };
      return callbacks;
    };
  }

  function patch_keyboard_actions() {
    log('Patching Jupyter up/down actions');

    var kbm = Jupyter.keyboard_manager;

    var action_up = kbm.actions.get(kbm.command_shortcuts.get_shortcut('up'));
    action_up.handler = function (env) {
      for (var index = env.notebook.get_selected_index() - 1; (index !== null) && (index >= 0); index--) {
        if (env.notebook.get_cell(index).element.is(':visible')) {
          env.notebook.select(index);
          env.notebook.focus_cell();
          return;
        }
      }
    };

    var action_down = kbm.actions.get(kbm.command_shortcuts.get_shortcut('down'));
    action_down.handler = function (env) {
      var ncells = env.notebook.ncells();
      for (var index = env.notebook.get_selected_index() + 1; (index !== null) && (index < ncells); index++) {
        if (env.notebook.get_cell(index).element.is(':visible')) {
          env.notebook.select(index);
          env.notebook.focus_cell();
          return;
        }
      }
    };

    var action_run_select_below = kbm.actions.get(kbm.command_shortcuts.get_shortcut('shift-enter'));
    action_run_select_below.handler = function (env) {
      var indices = env.notebook.get_selected_cells_indices();
      var cell_index;
      if (indices.length > 1) {
        env.notebook.execute_cells(indices);
        cell_index = Math.max.apply(Math, indices);
      } else {
        var cell = env.notebook.get_selected_cell();
        cell_index = env.notebook.find_cell_index(cell);
        cell.execute();
      }

      // If we are at the end always insert a new cell and return
      if (cell_index === (env.notebook.ncells()-1)) {
        env.notebook.command_mode();
        env.notebook.insert_cell_below();
        env.notebook.select(cell_index+1);
        env.notebook.edit_mode();
        env.notebook.scroll_to_bottom();
        env.notebook.set_dirty(true);
        return;
      }

      env.notebook.command_mode();
      if (env.notebook.get_cell(cell_index+1).element.is(':visible')) {
        env.notebook.select(cell_index + 1);
      } else {
        env.notebook.insert_cell_below();
        env.notebook.select(cell_index + 1);
        env.notebook.edit_mode();

      }
      env.notebook.focus_cell();
      env.notebook.set_dirty(true);
    };
  }


  return {
    TOCView : TOCView
  };

});
