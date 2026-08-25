Maintenance guide
==================

API documentation is generated from the modules listed in the API pages. Add
public classes and functions to the relevant ``automodule`` page and provide
Google-style docstrings with explicit ``Args``, ``Returns``, ``Raises``, and
side effects.

When changing a data contract, update:

#. the YAML examples in ``docs/user-guide/configuration.rst``;
#. the workbook schema in ``docs/user-guide/reports.rst``;
#. the relevant Mermaid workflow/class diagram;
#. tests under ``tests/``;
#. the technical protocol when the research recommendation changes.

Build locally with warnings treated as errors. The Pages workflow also checks
all documentation changes on pull requests before deployment.
