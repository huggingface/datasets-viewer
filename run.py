import streamlit as st
import numpy as np
import pandas as pd
import nlp
from dataclasses import asdict
import yaml
import textwrap
import tornado
import json

from streamlit.server.Server import Server

import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server





MAX_SIZE = 40000000000


## Hack to extend the width of the main pane.
def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    th {{
        text-align: left;
        font-size: 110%;
       

     }}

    tr:hover {{
        background-color: #ffff99;
        }}

    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

import time



app_state = st.experimental_get_query_string()

start = True
loaded = False
if app_state == "NOT_INITIALIZED":
    latest_iteration = st.empty()
    bar = st.progress(0)
    start = False
    for i in range(0, 101, 10):
      # Update the progress bar with each iteration.
      #latest_iteration.text(f'Iteration {i+1}')
      bar.progress(i)
      time.sleep(0.1)
      if i == 100:
          start = True
          bar.empty()
          loaded = True

if start:
    app_state = st.experimental_get_query_string()
    print("appstate is", app_state)
    app_state.setdefault("dataset", "glue")
    if len(app_state.get("dataset", [])) == 1:
        app_state["dataset"] = app_state["dataset"][0]
    if len(app_state.get("config", [])) == 1:
        app_state["config"] = app_state["config"][0]

    INITIAL_SELECTION = app_state["dataset"]
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
            builder_instance.manual_download_instructions is None
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
    for i, dataset in enumerate(nlp.list_datasets(with_community_datasets=False)):
        if dataset.id == INITIAL_SELECTION:
            start = i
        datasets.append(dataset)

    option = st.sidebar.selectbox(
        "Dataset", datasets, index=start, format_func=lambda a: a.id
    )
    print(option.id)
    app_state["dataset"] = option.id
    st.experimental_set_query_string(app_state)    

    # Side bar Configurations.
    configs = get_confs(option.id)
    conf_avail = len(configs) > 0
    conf_option = None
    if conf_avail:
        start = 0
        for i, conf in enumerate(configs):
            if conf.name == app_state.get("config", None):
                start = i 
        conf_option = st.sidebar.selectbox("Subset", configs, index=start, format_func=lambda a: a.name)
        app_state["config"] = conf_option.name
    
    else:
        if "config" in app_state:
            del app_state["config"]
    st.experimental_set_query_string(app_state)    

    dts, fail = get(str(option.id), str(conf_option.name) if conf_option else None)


    # Main panel setup.
    if fail:
        st.markdown(
            "Dataset is too large to browse or requires manual download. Check it out in the nlp library! \n\n Size: "
            + str(dts.info.size_in_bytes) + "\n\n Instructions: " + str(dts.manual_download_instructions)
        )
    else:

        k = list(dts.keys())
        index = 0
        if "train" in dts.keys():
            index = k.index("train")
        split = st.sidebar.selectbox("Split", k, index=index)

        d = dts[split]

        keys = list(d[0].keys())

        st.subheader(
            "Dataset: "
            + option.id
            + " "
            + (("/ " + conf_option.name) if conf_option else "")
            + " - " + d.info.homepage
        )

        md = """
        %s
        """ % (
            option.description.replace("\\", "") if option.description else ""
        )
        st.markdown(md)


        
        step = 50
        offset = st.sidebar.number_input(
            "Offset (Size: %d)"%len(d), min_value=0, max_value=int(len(d)) - step, value=0, step=step
        )

        citation = st.sidebar.checkbox("Show Citations", False)
        table = not st.sidebar.checkbox("Show List View", False)
        show_features = st.sidebar.checkbox("Show Features", False)
        md = """
```
%s
```
""" % (
            d.info.citation.replace("\\", "").replace("}", " }").replace("{", "{ "),
        )
        if citation:
            st.markdown(md)        
        #st.text("Features:")
        if show_features:
            on_keys = st.multiselect("Features", keys, keys)
            st.write(d.features)
        else:
            on_keys = keys
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
                        df_item[k] = textwrap.fill(out, width=50)
                    elif isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
                        df_item[k] = v
                    else:
                        out = json.dumps(v, indent=2, sort_keys=True)
                        df_item[k] = out
                df.append(df_item)
            df2 = df
            df = pd.DataFrame(df).set_index("_number")
            def hover(hover_color="#ffff99"):
                return dict(selector="tr:hover",
                            props=[("background-color", "%s" % hover_color)])

            styles = [
                hover(),
                dict(selector="th", props=[("font-size", "150%"),
                                           ("text-align", "center")]),
                dict(selector="caption", props=[("caption-side", "bottom")])
            ]

            # Table view. Use pands styling.
            style = df.style.set_properties(**{"text-align": "left", "white-space":"pre"}) \
                            .set_table_styles(
                                [dict(selector="th", props=[("text-align", "left")])]
                            )
            style = style.set_table_styles(styles)
            st.table(style)


    # Additional dataset installation and sidebar properties.
    md = """
    ### Code

    ```python
    !pip install nlp
    from nlp import load_dataset
    dataset = load_dataset(
       '%s'%s)
    ```

    """ % (
        option.id,
        (", '" + conf_option.name + "'") if conf_option else "",
    )
    st.sidebar.markdown(md)





