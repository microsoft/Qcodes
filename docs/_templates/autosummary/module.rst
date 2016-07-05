{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
   {% for item in classes %}
      {{ item }}
        {% block methods %}
        {% if methods %}

        ..
           HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
           .. autosummary::
              :toctree:
              {% for item in all_methods %}
              {%- if not item.startswith('_') or item in ['__call__'] %}
              {{ name }}.{{ item }}
              {%- endif -%}
              {%- endfor %}

        {% endif %}
        {% endblock %}

        {% block attributes %}
        {% if attributes %}

        ..
           HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
           .. autosummary::
              :toctree:
              {% for item in all_attributes %}
              {%- if not item.startswith('_') %}
              {{ name }}.{{ item }}
              {%- endif -%}
              {%- endfor %}

        {% endif %}
        {% endblock %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
