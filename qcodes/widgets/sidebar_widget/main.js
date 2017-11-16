// Adapted from https://gist.github.com/magican/5574556
// by minrk https://github.com/minrk/ipython_extensions
// See the history of contributions in README.md

define([
  'require',
  'jquery',
  'base/js/namespace',
  'notebook/js/codecell',
  './sidebar_toc'
], function(
  require,
  $,
  IPython,
  codecell,
  sidebar_toc
) {
  "use strict";

// ...........Parameters configuration......................
  // default values for system-wide configurable parameters
  var cfg={'threshold':4,
    'navigate_menu':true,
    'moveMenuLeft': true,
    'widenNotebook': false,
    collapse_to_match_collapsible_headings: false,
  };
  // default values for per-notebook configurable parameters
  var metadata_settings = {
    nav_menu: {},
    number_sections: true,
    sideBar: true,
    skip_h1_title: false,
    toc_position: {},
    toc_section_display: 'block',
    toc_window_display: false,
    hide_others: true
  };
  // add per-notebook settings into global config object
  $.extend(true, cfg, metadata_settings);

//.....................global variables....
  var st={};
  st.oldTocHeight = undefined;
  st.toc_index=0;

});
