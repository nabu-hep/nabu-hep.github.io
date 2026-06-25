####################
The ``.nabu`` format
####################

This page documents the on-disk file format that ``nabu`` uses to save and load a
trained :class:`~nabu.Likelihood`. The format is produced by
:meth:`nabu.Likelihood.save` and consumed by :meth:`nabu.Likelihood.load`,
both defined in :mod:`nabu.likelihood`.

A ``.nabu`` file stores *two distinct things*:

#. a **human-readable JSON header** that fully describes the *architecture* of
   the model (which flow, which hyper-parameters, which posterior transform),
   and
#. a **binary payload** that holds the *numerical leaves* (trained weights and
   parameters) of the model.

This split is deliberate: the header is everything required to rebuild an
*untrained* skeleton of the model, and the binary payload is everything
required to populate that skeleton with the values learned during training.


Overall layout
==============

A ``.nabu`` file is a single binary file with the following structure::

    <JSON header on one line>\n
    <equinox-serialised binary payload>

That is:

* The **first line** of the file is a UTF-8 encoded JSON object, terminated by
  a single newline character (``\n``).
* **Everything after** the first newline is an opaque binary blob written by
  :func:`equinox.tree_serialise_leaves`.

Because the header is always exactly one line, a reader can recover it with a
single ``readline()`` and treat the remainder of the stream as the binary
payload. This is precisely what :meth:`~nabu.Likelihood.load` does.

The file name is normalised on save: any extension supplied by the user is
replaced with ``.nabu`` (see :meth:`~nabu.Likelihood.save`). On load the suffix
is asserted to be ``.nabu``.


The JSON header
===============

The header is produced by :meth:`nabu.Likelihood.to_spec`, which builds a
validated :class:`nabu.serialization.LikelihoodSpec`, and written by
:meth:`~nabu.Likelihood.save` through that spec's ``to_dict`` method. It has the
following top-level structure:

.. code-block:: json

    {
      "model_type": "flow",
      "model": { "...": "..." },
      "posterior_transform": { "...": "..." },
      "version": "0.0.1",
      "format_version": 1
    }

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Description
   * - ``model_type``
     - String identifier of the likelihood subclass. For
       :class:`~nabu.flow.FlowLikelihood` this is ``"flow"``. Only ``"flow"`` is
       currently supported; :class:`~nabu.serialization.LikelihoodSpec` rejects
       any other value.
   * - ``model``
     - Architecture description of the model, produced by the subclass'
       :meth:`~nabu.Likelihood.to_dict` and validated against the corresponding
       flow specification on load. See `The model block`_.
   * - ``posterior_transform``
     - Description of the posterior transform, produced by
       :meth:`nabu.transform_base.PosteriorTransform.to_dict`. See
       `The posterior_transform block`_.
   * - ``version``
     - The installed version of the ``nabu-hep`` package at save time, taken
       from ``importlib.metadata.version("nabu-hep")``. Purely diagnostic; see
       `Schema versioning`_.
   * - ``format_version``
     - Integer version of the header schema itself, distinct from ``version``.
       The current schema version is ``1``; a header that omits this field is
       read as version ``1``. See `Schema versioning`_.

The header is serialised with :class:`nabu.likelihood.NumpyEncoder`, a custom
:class:`json.JSONEncoder` that converts NumPy scalar and array types to native
JSON. The conversions are:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - NumPy type
     - JSON representation
   * - integer types (``np.int8`` … ``np.uint64``)
     - JSON number (``int``)
   * - floating types (``np.float16`` … ``np.float64``)
     - JSON number (``float``)
   * - complex types (``np.complex64``, ``np.complex128``)
     - object ``{"real": ..., "imag": ...}``
   * - ``np.ndarray``
     - JSON array (via ``ndarray.tolist()``)
   * - ``np.bool_``
     - JSON boolean
   * - ``np.void``
     - ``null``


The ``model`` block
-------------------

For a :class:`~nabu.flow.FlowLikelihood`, ``to_dict()`` returns the metadata
dictionary that was captured when the flow was constructed. It is a
**single-key object**: the key is the name of the flow constructor and the
value is the dictionary of keyword arguments needed to rebuild it.

.. code-block:: json

    {
      "masked_autoregressive_flow": {
        "dim": 2,
        "transformer": null,
        "cond_dim": null,
        "flow_layers": 8,
        "nn_width": 50,
        "nn_depth": 1,
        "activation": "relu",
        "permutation": "reversed",
        "random_seed": 0
      }
    }

The recognised flow constructor names are exactly the keys of the flow registry
in :mod:`nabu.flow` (resolved on load by ``get_flow``):

* ``masked_autoregressive_flow``
* ``coupling_flow``
* ``block_neural_autoregressive_flow``
* ``planar_flow``
* ``triangular_spline_flow``
* any custom flow added via :func:`nabu.flow.register_flow`

The argument dictionary is built automatically by the ``serialise_wrapper``
decorator (see :mod:`nabu.flow._serialisation_utils`). It introspects the
constructor signature and records each argument's actual value, with two
notable rules:

* The ``self`` and ``key`` parameters are skipped.
* Arguments are stored verbatim when they are JSON-friendly scalars or
  sequences (``int``, ``float``, ``str``, ``bool``, ``list``, ``tuple``,
  arrays); other callables/classes are recursively serialised by name.
* **Lambda functions are not supported** and raise
  ``UnsupportedMethod`` at serialisation time.

Nested transformer bijections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some flows accept a ``transformer`` argument, which is itself a bijection. When
present it is serialised in the same single-key form, mapping a bijector name
to its keyword arguments:

.. code-block:: json

    {
      "coupling_flow": {
        "dim": 4,
        "transformer": {
          "RationalQuadraticSpline": {
            "knots": 8,
            "interval": 4
          }
        },
        "flow_layers": 8,
        "nn_width": 50,
        "activation": "relu",
        "permutation": "reversed",
        "random_seed": 0
      }
    }

The bijector name must be one of the keys registered in
:mod:`nabu.flow._bijectors` (resolved on load by ``get_bijector``), e.g.
``Affine``, ``RationalQuadraticSpline``, ``Coupling``, ``Tanh``,
``TriangularAffine``, etc. The full list is available from
:func:`nabu.flow.available_bijectors`.

.. note::

   On load, any list-valued transformer argument is converted back to a
   ``tuple`` before the bijector is instantiated, because several ``flowjax``
   bijections expect tuple-valued shape arguments.


The ``posterior_transform`` block
---------------------------------

This block is produced by
:meth:`nabu.transform_base.PosteriorTransform.to_dict` and describes the
invertible transform applied to the data before training (forward) and to the
samples after sampling (backward). It takes one of three shapes.

**No transform** (identity) — an empty object:

.. code-block:: json

    {}

**Shift–scale** transform:

.. code-block:: json

    {
      "shift-scale": {
        "shift": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "log_axes": null,
        "log_shift": null,
        "log_scale": null
      }
    }

where ``shift`` and ``scale`` are per-feature lists. The optional ``log_axes``,
``log_shift`` and ``log_scale`` entries support taking the logarithm of
selected axes before the shift/scale is applied; they are ``null`` when unused.

**Dalitz** transform (Dalitz coordinates ↔ unit-square coordinates):

.. code-block:: json

    {
      "dalitz": {
        "md": 1.86483,
        "ma": 0.13957,
        "mb": 0.13957,
        "mc": 0.13957
      }
    }

where ``md`` is the mother-particle mass and ``ma``, ``mb``, ``mc`` are the
three daughter-particle masses.

.. warning::

   A ``user-defined`` posterior transform cannot be serialised: its
   ``forward``/``backward`` callables have no JSON representation. Saving a
   likelihood whose posterior transform is ``user-defined`` therefore **raises**
   a :class:`~nabu.serialization.SerializationError`, rather than silently
   degrading to the identity transform as earlier versions did. Re-apply the
   user-defined transform manually after loading a model trained with one.


The binary payload
==================

Everything after the header's terminating newline is written by
:func:`equinox.tree_serialise_leaves(f, self.model)`. This payload contains
**only the array leaves** of the model pytree (the trained weights, biases,
spline parameters, etc.) in equinox's native binary serialisation format. It
does **not** contain the pytree *structure* — that information lives entirely
in the JSON header.

This is why loading is a two-step process (see below): the structure is rebuilt
from the header, and only then are the leaves read back into it.


Reading and writing
===================

Writing (:meth:`nabu.Likelihood.save`)
--------------------------------------

#. Normalise the file name to carry a ``.nabu`` suffix.
#. Build a validated :class:`nabu.serialization.LikelihoodSpec` via
   :meth:`~nabu.Likelihood.to_spec` and render the header with its ``to_dict``
   method (this includes both ``version`` and ``format_version``).
#. Open the target file in binary write mode.
#. Write ``json.dumps(header, cls=NumpyEncoder) + "\n"`` as UTF-8 bytes.
#. Append the model leaves with
   :func:`equinox.tree_serialise_leaves(f, self.model)`.

Reading (:meth:`nabu.Likelihood.load`)
--------------------------------------

#. Check the file exists and has a ``.nabu`` suffix, raising
   :class:`nabu.serialization.SerializationError` otherwise.
#. Open the file in binary read mode.
#. Read the first line, decode it as UTF-8 and parse it as JSON
   (``json.loads(f.readline().decode())``).
#. Parse and validate the header into a
   :class:`nabu.serialization.LikelihoodSpec` via its ``from_dict`` method. A
   malformed header raises :class:`~nabu.serialization.SchemaError`; an unknown
   flow, bijector or transform name raises
   :class:`~nabu.serialization.UnknownTagError`.
#. Build an untrained flow skeleton from the model spec via
   ``spec.model.build(random_seed)``. This rebuilds any nested ``transformer``
   bijection (converting list arguments to tuples) and overrides the random
   seed with the ``random_seed`` argument passed to
   :meth:`~nabu.Likelihood.load`.
#. Restore the trained leaves into that skeleton with
   :func:`equinox.tree_deserialise_leaves`, reading the rest of the file.
#. Build the :class:`~nabu.transform_base.PosteriorTransform` from the
   posterior-transform spec via ``spec.posterior_transform.build()`` (an empty
   object yields an identity transform).

.. note::

   The ``random_seed`` only affects the initial parameters of the rebuilt
   skeleton; these are immediately overwritten by the deserialised leaves, so
   the reconstructed model is deterministic and independent of the seed. The
   seed matters only if deserialisation of the leaves is skipped.

.. important::

   Loading requires that the JSON architecture in the header can rebuild a
   pytree whose structure matches the saved leaves. If a flow constructor's
   signature changes incompatibly between the save-time and load-time versions
   of ``nabu``/``flowjax``, the leaves may fail to map back onto the rebuilt
   skeleton. The ``version`` field in the header records the ``nabu-hep``
   version that produced the file and is intended to aid such diagnostics.


Schema versioning
=================

The header carries two independent version fields:

* ``version`` records the ``nabu-hep`` package version at save time, taken from
  ``importlib.metadata.version("nabu-hep")``. It is purely diagnostic and is not
  checked on load.
* ``format_version`` is an integer version of the *header schema itself*. The
  current schema version is ``1``. A header that omits ``format_version`` — i.e.
  a file written before the field was introduced — is read as version ``1``,
  preserving backward compatibility with older ``.nabu`` files.

On load, a ``format_version`` greater than the version supported by the
installed ``nabu`` raises :class:`~nabu.serialization.SchemaError`, signalling
that the file was written by a newer release of the format.


Worked example
==============

A complete ``.nabu`` file for a two-dimensional masked autoregressive flow with
a shift–scale posterior transform looks like the following. The first line is
the JSON header (shown pretty-printed here; on disk it is a single line); the
remaining bytes are the opaque binary payload.

.. code-block:: text

    {"model_type": "flow", "model": {"masked_autoregressive_flow": {"dim": 2, "transformer": null, "cond_dim": null, "flow_layers": 8, "nn_width": 50, "nn_depth": 1, "activation": "relu", "permutation": "reversed", "random_seed": 0}}, "posterior_transform": {"shift-scale": {"shift": [0.0, 0.0], "scale": [1.0, 1.0], "log_axes": null, "log_shift": null, "log_scale": null}}, "version": "0.0.1", "format_version": 1}
    <binary equinox leaves follow this newline …>
