(require.specified('base/js/namespace') ? define : function (deps, callback) {
  // if here, the Jupyter namespace hasn't been specified to be loaded.
  // This means that we're probably embedded in a page, so we need to make
  // our definition with a specific module name
  return define('nbextensions/sidebar_toc/sidebar_toc', deps, callback);
})(['jquery', 'require'], function ($, require) {
  "use strict";

  var IPython;
  var events;
  var liveNotebook = false;

  function setNotebookWidth(cfg, st) {
    //cfg.widenNotebook  = true;
    if (cfg.sideBar) {
      if ($('#toc-wrapper').is(':visible')) {
        $('#notebook-container').css('margin-left', $('#toc-wrapper').width() + 30)
        $('#notebook-container').css('width', $('#notebook').width() - $('#toc-wrapper').width() - 30)
      } else {
        if (cfg.widenNotebook) {
          $('#notebook-container').css('margin-left', 30);
          $('#notebook-container').css('width', $('#notebook').width() - 30);
        } else { // original width
          $("#notebook-container").css({'width':''})
        }
      }
    } else {
      if (cfg.widenNotebook) {
        $('#notebook-container').css('margin-left', 30);
        $('#notebook-container').css('width', $('#notebook').width() - 30);
      } else { // original width
        $("#notebook-container").css({'width':''})
      }
    }
  }

  function setSideBarHeight(cfg, st) {
    if (cfg.sideBar) {
      var headerVisibleHeight = $('#header').is(':visible') ? $('#header').height() : 0
      $('#toc-wrapper').css('top', liveNotebook ? headerVisibleHeight : 0)
      $('#toc-wrapper').css('height', $('#site').height());
      $('#toc').css('height', $('#toc-wrapper').height() - $('#toc-header').height())
    }
  }

// Table of Contents =================================================================
  var table_of_contents = function (cfg,st) {

    var toc_wrapper = $("#toc-wrapper");
    // var toc_index=0;
    if (toc_wrapper.length === 0) {
      create_toc_div(cfg,st);
    }
    var segments = [];
    var ul = $("<ul/>").addClass("toc-item").attr('id','toc-level0');

    // update toc element
    $("#toc").empty().append(ul);

    $(window).resize(function(){
      $('#toc').css({maxHeight: $(window).height() - 30});
      $('#toc-wrapper').css({maxHeight: $(window).height() - 10});
      setSideBarHeight(cfg, st),
        setNotebookWidth(cfg, st);
    });

    $(window).trigger('resize');

  };


  var create_toc_div = function (cfg,st) {
    var toc_wrapper = $('<div id="toc-wrapper"/>')
      .append(
        $("<div/>").attr("id", "toc").addClass('toc')
      );

    var sidebar = $('<div id="sidebar-wrapper"/>').append(toc_wrapper);

    $("body").append(sidebar);

    // On header/menu/toolbar resize, resize the toc itself
    // (if displayed as a sidebar)
    if (liveNotebook) {
      $([Jupyter.events]).on("resize-header.Page", function() {setSideBarHeight(cfg, st);});
      $([Jupyter.events]).on("toggle-all-headers", function() {setSideBarHeight(cfg, st);});
    }


    $('#sidebar-wrapper').resizable({
      resize : function(event,ui){
        if (cfg.sideBar){
          setNotebookWidth(cfg, st)
        }
        else {
          $('#toc').css('height', $('#sidebar-wrapper').height()-$('#toc-header').height());
        }
      },
      start : function(event, ui) {
        $(this).width($(this).width());
        //$(this).css('position', 'fixed');
      },
      stop :  function (event,ui){ // on save, store toc position
        if(liveNotebook){
          IPython.notebook.metadata.toc['toc_position']={
            'left':$('#sidebar-wrapper').css('left'),
            'top':$('#sidebar-wrapper').css('top'),
            'height':$('#sidebar-wrapper').css('height'),
            'width':$('#sidebar-wrapper').css('width'),
            'right':$('#sidebar-wrapper').css('right')};
          $('#toc').css('height', $('#sidebar-wrapper').height()-$('#toc-header').height())
          IPython.notebook.set_dirty();
        }
        // Ensure position is fixed (again)
        //$(this).css('position', 'fixed');
      }
    })


    // restore toc position at load
    if(liveNotebook){
      if (IPython.notebook.metadata.toc['toc_position'] !== undefined){
        $('#sidebar-wrapper').css(IPython.notebook.metadata.toc['toc_position']);
      }
    }
    // Ensure position is fixed
    $('#sidebar-wrapper').css('position', 'fixed');

    // Restore toc display
    if(liveNotebook){
      if (IPython.notebook.metadata.toc !== undefined) {
        if (IPython.notebook.metadata.toc['toc_section_display']!==undefined)  {
          $('#toc').css('display',IPython.notebook.metadata.toc['toc_section_display'])
          $('#toc').css('height', $('#toc-wrapper').height()-$('#toc-header').height())
          if (IPython.notebook.metadata.toc['toc_section_display']=='none'){
            $('#toc-wrapper').addClass('closed');
            $('#toc-wrapper').css({height: 40});
            $('#toc-wrapper .hide-btn')
              .text('[+]')
              .attr('title', 'Show ToC');
          }
        }
        if (IPython.notebook.metadata.toc['toc_window_display']!==undefined)    {
          console.log("******Restoring toc display");
          $('#sidebar-wrapper').css('display',IPython.notebook.metadata.toc['toc_window_display'] ? 'block' : 'none');
        }
      }
    }

    // if toc-wrapper is undefined (first run(?), then hide it)
    if ($('#sidebar-wrapper').css('display')==undefined) $('#sidebar-wrapper').css('display',"none") //block
    //};

    $('#site').bind('siteHeight', function() {
      if (cfg.sideBar) $('#toc-wrapper').css('height',$('#site').height());})

    $('#site').trigger('siteHeight');

    // Initial style for sidebar
    $('#sidebar-wrapper').addClass('sidebar-wrapper');
    if (!liveNotebook) {
      $('#sidebar-wrapper').css('width', '202px');
      $('#notebook-container').css('margin-left', '212px');
      $('#sidebar-wrapper').css('height', '96%');
      $('#toc').css('height', $('#sidebar-wrapper').height() - $('#toc-header').height())
    } else {
      if (cfg.toc_window_display) {
        setTimeout(function() {
          setNotebookWidth(cfg, st)
        }, 500)
      }
      setTimeout(function() {
        $('#sidebar-wrapper').css('height', $('#site').height());
        $('#toc').css('height', $('#toc-wrapper').height() - $('#toc-header').height())
      }, 500)
    }
    setTimeout(function() { $('#sidebar-wrapper').css('top', liveNotebook ? $('#header').height() : 0); }, 500) //wait a bit
    $('#sidebar-wrapper').css('left', 0);

  }



  return {
    highlight_toc_item: highlight_toc_item,
    table_of_contents: table_of_contents,
    toggle_toc: toggle_toc,
  };
}
