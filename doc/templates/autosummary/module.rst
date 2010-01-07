{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   .. inheritance-diagram:: {{ fullname }}
      :parts: 1

   {% block functions %}
   {% if functions or methods %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% for item in methods %}
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
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
