import streamlit as st
import numpy as np
import pandas as pd
import datasets
from dataclasses import asdict
import yaml
import textwrap
import tornado
import json
import time
import sys
import glob
import json
#from st_annotated_text import annotated_text
nlp = datasets


MAX_SIZE = 40000000000
if len(sys.argv) > 1:
    path_to_datasets = sys.argv[1]
else:
    path_to_datasets = None

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


def render_features(features):
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, nlp.features.ClassLabel):
        return features.names

    if isinstance(features, nlp.features.Value):
        return features.dtype

    if isinstance(features, nlp.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


tag_dict = {}
data_tags = {}
    
for d in json.load(open("datasets.json", "r")):
    if "card_data" in d:
        data_tags[d["id"]] = d["card_data"]
        
        for k, v in d["card_data"].items():
            for value in v:
                tag = k + u":" + str(value)
                tag_dict.setdefault(tag, [])
                tag_dict[tag].append(d["id"])
    else:
        data_tags[d["id"]] = {}

app_state = st.experimental_get_query_params()
# print(app_state)
start = True
loaded = True
INITIAL_SELECTION = ""
# if app_state == "NOT_INITIALIZED":
#     latest_iteration = st.empty()
#     bar = st.progress(0)
#     start = False
#     for i in range(0, 101, 10):
#         # Update the progress bar with each iteration.
#         # latest_iteration.text(f'Iteration {i+1}')
#         bar.progress(i)
#         time.sleep(0.1)
#         if i == 100:
#             start = True
#             bar.empty()
#             loaded = True

#             app_state = st.experimental_get_query_params()
#             print("appstate is", app_state)
app_state.setdefault("dataset", "glue")
if len(app_state.get("dataset", [])) == 1:
    app_state["dataset"] = app_state["dataset"][0]
    INITIAL_SELECTION = app_state["dataset"]
# if len(app_state.get("config", [])) == 1:
#     app_state["config"] = app_state["config"][0]
print(INITIAL_SELECTION)

if start:
    ## Logo and sidebar decoration.
    st.sidebar.markdown(
        """<center>
    <a href="https://github.com/huggingface/datasets">
    </a>
    </center>""",
        unsafe_allow_html=True,
    )
    st.sidebar.image("datasets_logo_name.png", width=260)
    st.sidebar.markdown(
        "<center><h2><a href='https://github.com/huggingface/datasets'>github/huggingface/datasets</h2></a></center>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
    <center>
        <a target="_blank" href="https://huggingface.co/datasets/">Docs</a> | 
        <a target="_blank" href="https://colab.research.google.com/github/huggingface/datasets/blob/master/notebooks/Overview.ipynb"> Overview</a>
    | <a href="https://huggingface.co/datasets/add_dataset.html" target="_blank">Add Dataset</a>
    </center>""",
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("")

    ## Interaction with the nlp libary.
    @st.cache
    def get_confs(opt):
        "Get the list of confs for a dataset."
        if path_to_datasets is not None and opt is not None:
            path = path_to_datasets + opt
        else:
            path = opt

        module_path = nlp.load.prepare_module(path, dataset=True
        )
        # Get dataset builder class from the processing script
        builder_cls = nlp.load.import_main_class(module_path[0], dataset=True)
        # Instantiate the dataset builder
        confs = builder_cls.BUILDER_CONFIGS
        if confs and len(confs) > 1:
            return confs
        else:
            return []

    @st.cache(allow_output_mutation=True)
    def get(opt, conf=None):
        "Get a dataset from name and conf"
        if path_to_datasets is not None:
            path = path_to_datasets + opt
        else:
            path = opt
        
        module_path = nlp.load.prepare_module(path, dataset=True)
        builder_cls = nlp.load.import_main_class(module_path[0], dataset=True)
        if conf:
            builder_instance = builder_cls(name=conf, cache_dir=path if path_to_datasets is not None else None)
        else:
            builder_instance = builder_cls(cache_dir=path if path_to_datasets is not None else None)
        fail = False
        if path_to_datasets is not None:
            dts = nlp.load_dataset(path,
                                   name=builder_cls.BUILDER_CONFIGS[0].name if builder_cls.BUILDER_CONFIGS else None,
            )
            dataset = dts

        elif (
            builder_instance.manual_download_instructions is None
            and builder_instance.info.size_in_bytes is not None
            and builder_instance.info.size_in_bytes < MAX_SIZE):
            builder_instance.download_and_prepare()
            dts = builder_instance.as_dataset()
            dataset = dts
        else:
            dataset = builder_instance
            fail = True
        return dataset, fail

    tags = list(tag_dict.keys())
    
    tags.sort()
    dataset_filter = st.sidebar.multiselect(
        "Filter by Tags", tags
    )

    
    # Dataset select box.
    datasets = []
    selection = None


    if path_to_datasets is None:
        list_of_datasets = nlp.list_datasets(with_community_datasets=False)
    else:
        list_of_datasets = sorted(glob.glob(path_to_datasets + "*"))

        
    if dataset_filter:
        list_of_datasets = set(list_of_datasets)
        for f in dataset_filter:
            list_of_datasets = list_of_datasets & set(tag_dict[f])
        list_of_datasets = list(list_of_datasets)
        list_of_datasets.sort()
        
    print(list_of_datasets)
    for i, dataset in enumerate(list_of_datasets):
        dataset = dataset.split("/")[-1]
        if INITIAL_SELECTION and dataset == INITIAL_SELECTION:
            selection = i
        datasets.append(dataset )

    if selection is not None:
        option = st.sidebar.selectbox(
            "Dataset (Size: %d)"%(len(datasets)), datasets, index=selection, format_func=lambda a: a
        )
    else:
        option = st.sidebar.selectbox("Dataset (Size: %d)"%(len(datasets)), datasets, format_func=lambda a: a)
    print(option)
    app_state["dataset"] = option
    # st.experimental_set_query_params(**app_state)

    # Side bar Configurations.
    configs = get_confs(option)
    conf_avail = len(configs) > 0
    conf_option = None
    if conf_avail:
        start = 0
        for i, conf in enumerate(configs):
            if conf.name == app_state.get("config", None):
                start = i
        conf_option = st.sidebar.selectbox(
            "Subset", configs, index=start, format_func=lambda a: a.name
        )
        app_state["config"] = conf_option.name

    else:
        if "config" in app_state:
            del app_state["config"]
    # st.experimental_set_query_params(**app_state)

    dts, fail = get(str(option), str(conf_option.name) if conf_option else None)

    # Main panel setup.
    if fail:
        st.markdown(
            "Dataset is too large to browse or requires manual download. Check it out in the datasets library! \n\n Size: "
            + str(dts.info.size_in_bytes)
            + "\n\n Instructions: "
            + str(dts.manual_download_instructions)
        )
    else:

        k = list(dts.keys())
        index = 0
        if "train" in dts.keys():
            index = k.index("train")
        split = st.sidebar.selectbox("Split", k, index=index)

        d = dts[split]

        keys = list(d[0].keys())

        st.header(
            "Dataset: "
            + option
            + " "
            + (("/ " + conf_option.name) if conf_option else "")
        )
        s = []
        for k, v in app_state.items():
            s .append( k + "=" + v)

        st.markdown(
            "*Homepage*: "
            + d.info.homepage
            + "\n\n*Dataset*: https://github.com/huggingface/datasets/blob/master/datasets/%s/%s.py"
            % (option, option)
        )
        st.markdown(
            "*Permalink*: https://huggingface.co/datasets/viewer/?"
            + "&".join(s)
        )

        md = """
        %s
        """ % (
            d.info.description.replace("\\", "") if option else ""
        )
        st.markdown(md)

        if option in data_tags and data_tags[option]:
            tags = data_tags[option]
            construct = "*Tags* \n"
            for k in tags:
                construct += f"* {k} : "
                construct += ", ".join([v for v in tags[k]])
                construct += "\n"
            st.markdown(construct)

        
        step = 50
        offset = st.sidebar.number_input(
            "Offset (Size: %d)" % len(d),
            min_value=0,
            max_value=int(len(d)) - step,
            value=0,
            step=step,
        )

        citation = st.sidebar.checkbox("Show Citations", False)
        table = not st.sidebar.checkbox("Show List View", False)
        show_features = st.sidebar.checkbox("Show Features", True)
        md = """
```
%s
```
""" % (
            d.info.citation.replace("\\", "").replace("}", " }").replace("{", "{ "),
        )
        if citation:
            st.markdown(md)
        # st.text("Features:")
        if show_features:
            on_keys = st.multiselect("Features", keys, keys)
            st.write(render_features(d.features))
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
                    elif (
                        isinstance(v, bool)
                        or isinstance(v, int)
                        or isinstance(v, float)
                    ):
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
                    elif (
                        isinstance(v, bool)
                        or isinstance(v, int)
                        or isinstance(v, float)
                    ):
                        df_item[k] = v
                    else:
                        out = json.dumps(v, indent=2, sort_keys=True)
                        df_item[k] = out
                df.append(df_item)
            df2 = df
            df = pd.DataFrame(df).set_index("_number")

            def hover(hover_color="#ffff99"):
                return dict(
                    selector="tr:hover",
                    props=[("background-color", "%s" % hover_color)],
                )

            styles = [
                hover(),
                dict(
                    selector="th",
                    props=[("font-size", "150%"), ("text-align", "center")],
                ),
                dict(selector="caption", props=[("caption-side", "bottom")]),
            ]

            # Table view. Use pands styling.
            style = df.style.set_properties(
                **{"text-align": "left", "white-space": "pre"}
            ).set_table_styles([dict(selector="th", props=[("text-align", "left")])])
            style = style.set_table_styles(styles)
            st.table(style)

    # Additional dataset installation and sidebar properties.
    md = """
    ### Code

    ```python
    !pip install datasets
    from datasets import load_dataset
    dataset = load_dataset(
       '%s'%s)
    ```

    """ % (
        option,
        (", '" + conf_option.name + "'") if conf_option else "",
    )
    st.sidebar.markdown(md)
