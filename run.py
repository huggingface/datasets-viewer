import streamlit as st
import numpy as np
import pandas as pd
import nlp
from dataclasses import asdict
import yaml
import textwrap

MAX_SIZE = 40000000000
INITIAL_SELECTION = "glue"

## Hack to extend the width of the main pane.
def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()


## Logo and sidebar decoration.
st.sidebar.markdown(
    """<center>
<a href="http://github.com/huggingface/nlp">
<img src="https://raw.githubusercontent.com/huggingface/nlp/master/docs/source/imgs/nlp_logo_name.png" width="200"></a>
</center>""",
    unsafe_allow_html=True,
)
st.sidebar.subheader("http://github.com/huggingface/nlp")
st.sidebar.markdown(
    """
<center>
    <a target="_blank" href="https://colab.research.google.com/github/huggingface/nlp/blob/master/notebooks/Overview.ipynb">Library Overview</a>
| <a href="https://github.com/huggingface/nlp/blob/master/CONTRIBUTING.md#how-to-add-a-dataset" target="_blank">Submit dataset</a>
</center>""",
    unsafe_allow_html=True,
)
st.sidebar.subheader("")


## Interaction with the nlp libary.
@st.cache
def get_confs(opt):
    "Get the list of confs for a dataset."
    module_path = nlp.load.prepare_module(opt, dataset=True)
    # Get dataset builder class from the processing script
    builder_cls = nlp.load.import_main_class(module_path, dataset=True)
    # Instantiate the dataset builder
    confs = builder_cls.BUILDER_CONFIGS
    if confs and len(confs) > 1:
        return confs
    else:
        return []

@st.cache(allow_output_mutation=True)
def get(opt, conf=None):
    "Get a dataset from name and conf"
    module_path = nlp.load.prepare_module(opt, dataset=True)
    builder_cls = nlp.load.import_main_class(module_path, dataset=True)
    if conf:
        builder_instance = builder_cls(name=conf)
    else:
        builder_instance = builder_cls()
    fail = False
    if (
        not builder_instance.does_require_manual_download
        and builder_instance.info.size_in_bytes is not None
        and builder_instance.info.size_in_bytes < MAX_SIZE
    ):
        builder_instance.download_and_prepare()
        dts = builder_instance.as_dataset()
        dataset = dts
    else:
        dataset = builder_instance
        fail = True
    return dataset, fail


# Dataset select box.
datasets = []
start = 0
for i, dataset in enumerate(nlp.list_datasets()):
    if dataset.id == INITIAL_SELECTION:
        start = i
    datasets.append(dataset)

option = st.sidebar.selectbox(
    "Dataset", datasets, index=start, format_func=lambda a: a.id
)

# Side bar Configurations.
configs = get_confs(option.id)
conf_avail = len(configs) > 0
conf_option = None
if conf_avail:
    conf_option = st.sidebar.selectbox("Subset", configs, format_func=lambda a: a.name)

table = not st.sidebar.checkbox("Full View", False)
dts, fail = get(str(option.id), str(conf_option.name) if conf_option else None)


# Main panel setup.
if fail:
    st.subheader(
        "Dataset is too large to browse or requires manual download. Check it out in the nlp library! Size: "
        + str(dts.info.size_in_bytes)
    )
else:
    st.subheader(
        "Dataset: "
        + option.id
        + " "
        + (("/" + conf_option.name) if conf_option else "")
    )
    k = list(dts.keys())
    index = 0
    if "train" in dts.keys():
        index = k.index("train")
    split = st.selectbox("Split", k, index=index)

    d = dts[split]

    keys = list(d[0].keys())
    on_keys = st.sidebar.multiselect("Keys", keys, keys)

    step = 50
    offset = st.number_input(
        "Offset", min_value=0, max_value=int(len(d)) - step, value=0, step=step
    )

    if not table:
        # Full view.
        for item in range(offset, offset + step):
            st.text("        ")
            st.text("                  ----  #" + str(item))
            st.text("        ")
            # Use st to write out.
            for k in on_keys:
                v = d[item][k]
                st.subheader(k)
                if isinstance(v, str):
                    out = v
                    st.text(textwrap.fill(out, width=120))
                elif isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
                    st.text(v)
                else:
                    st.write(v)

    else:
        # Table view. Use Pandas.
        df = []
        for item in range(offset, offset + step):
            df_item = {}
            df_item["_number"] = item
            for k in on_keys:
                v = d[item][k]
                if isinstance(v, str):
                    out = v
                    df_item[k] = textwrap.fill(out, width=80)
                elif isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
                    df_item[k] = v
                else:
                    df_item[k] = v
            df.append(df_item)

        df = pd.DataFrame(df).set_index("_number")

        # Table view. Use pands styling.
        style = df.style.set_properties(**{"text-align": "left"})
        style = style.set_table_styles(
            [dict(selector="th", props=[("text-align", "left")])]
        )
        st.table(style)


# Additional dataset installation and sidebar properties.
md = """
### Overview

```python
!pip install nlp
from nlp import load_dataset
dataset = load_dataset('%s'%s)
```

""" % (
    option.id,
    (", '" + conf_option.name + "'") if conf_option else "",
)
st.sidebar.markdown(md)

md = """
%s
""" % (
    option.description.replace("\\", "") if option.description else ""
)
st.sidebar.markdown(md)

if not fail:
    md = """
%s

%s

### Citation

```
%s
```

""" % (
        d.info.homepage,
        d.info.license,
        d.info.citation.replace("\\", "").replace("}", " }").replace("{", "{ "),
    )
    st.sidebar.markdown(md)
