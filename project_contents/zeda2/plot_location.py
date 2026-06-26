# ZELDA: Zalo Exploratory Location Data Analysys

import numpy as np
import folium
import pandas as pd
import folium.plugins as foliumplugins


from zeda2 import data_format_helper as _data_format_helper


##################################################################################################################################
################################# SCATTER ########################################################################################
##################################################################################################################################


def defaultplotDot(lat, lon, tooltip=None):
    if tooltip is not None:
        return folium.CircleMarker(location=[lat, lon], radius=2, weight=3, tooltip=tooltip)
    else:
        return folium.CircleMarker(location=[lat, lon], radius=2, weight=3)
    
    
def visualizeLocationScatter(df_input, coord_col = ['lat', 'lon'], plotfunction=defaultplotDot, tooltip_col=None):
    
    """
        Plot Scatter plot
        Arguments:
            df_input: data frame has column need plot
            coord_col: coordinate columns. Default: ['lat', 'lon']
            plotfunction: define points. default: 
            
                    def defaultplotDot(lat, lon, tooltip=None):
                        if tooltip is not None:
                            return folium.CircleMarker(location=[lat, lon], radius=2, weight=3, tooltip=tooltip)
                        else:
                            return folium.CircleMarker(location=[lat, lon], radius=2, weight=3)
        
        
            tooltip_col: Column of tooltip (diplayed on hovering on points). Default: None.
    """
 
    using_tooltip = tooltip_col is not None
    
    this_map = folium.Map(prefer_canvas=True)


    if using_tooltip:
        df_input.apply(lambda x : plotfunction(x[coord_col[0]], x[coord_col[1]], x[tooltip_col]).add_to(this_map), axis = 1)
    else:
        df_input.apply(lambda x : plotfunction(x[coord_col[0]], x[coord_col[1]], None).add_to(this_map), axis = 1)

    this_map.fit_bounds(this_map.get_bounds())

    return this_map

##################################################################################################################################
################################# HEATMAP ########################################################################################
##################################################################################################################################


def visualizeLocationHeatmap(df_input, coord_col = ['lat', 'lon'], radius = 25, hasweight = False, weight_col = None):
    """
        Plot heatmap
        Arguments:
            df_input: data frame has column need plot
            coord_col: coordinate columns. Default: ['lat', 'lon']
            hasweight: Is using weight (Default: None, This will be automatically turned on if: (1) len of coord_col is 3 of (2) weight_col is not none)
            weight_col: Column of weight, default: None, if hasweight is True and length of coord_col is less than 3, then weight_col is automatically set to "weight" (if the dataframe does not contain this column, error will be thrown)
    """
    
    using_weight = hasweight
    if len(coord_col) == 3:
        weight_col = coord_col[2]
        using_weight = True
    
    if weight_col is not None:
        using_weight = True

    if using_weight and weight_col is None:
        weight_col = 'weight'
        
    
    this_map = folium.Map(prefer_canvas=True)
    
    if using_weight:
        temp = df_input[list(set(coord_col + [weight_col]))]
        heat_data = [[row[coord_col[0]],row[coord_col[1]], row[weight_col]] for index, row in temp.iterrows()]
        foliumplugins.HeatMap(heat_data, radius = radius).add_to(this_map)
    else:
        temp = df_input[coord_col]
        heat_data = [[row[coord_col[0]],row[coord_col[1]]] for index, row in temp.iterrows()]
        foliumplugins.HeatMap(heat_data, radius = radius).add_to(this_map)


    #Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())
    return this_map


##################################################################################################################################
################################# CHOROPLETH #####################################################################################
##################################################################################################################################

def visualizeLocationChoroplethVietnam(df_input, data_col, data_desc, aaid_col = 'aaid', hover_config = None):
    
    """
        Plot Choropleth plot
        Arguments:
            df_input: data frame has column need plot
            data_col: name of column that contain data
            data_desc: Description of visualized data 
            aaid_col: admin area id column
            hover_config: Config of hover, Default: None.
                Format of hover_config: ([list of label, list of value]). 
                Eg. (['AdminID: ','User count: '], ['vng_id','count'])
    """

    default_column_exist = False
    
    if 'vng_id' in df_input.columns:
        default_column_exist = True
    
    if not default_column_exist:
        df_input['vng_id'] = df_input[aaid_col]
    
    vn_gpd = _data_format_helper.getVietnamGeoJson()
    
    temp_vn_gpd = pd.merge(vn_gpd, df_input, on='vng_id', how='inner')
    
    m = folium.Map()
    folium.Choropleth(
        geo_data=temp_vn_gpd,
        name="choropleth",
        data=df_input,
        columns=["vng_id", data_col],
        fill_color="YlGn",
        fill_opacity=0.5,
        line_opacity=0.2,
        legend_name=data_desc,
        key_on="feature.properties.vng_id",
    ).add_to(m)

    if hover_config is not None:

        style_function = lambda x: {'fillColor': '#ffffff', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.1, 
                                    'weight': 0.1}
        highlight_function = lambda x: {'fillColor': '#000000', 
                                        'color':'#000000', 
                                        'fillOpacity': 0.50, 
                                        'weight': 0.1}
        hover_layer = folium.features.GeoJson(
            temp_vn_gpd,
            style_function=style_function, 
            control=False,
            highlight_function=highlight_function, 
            tooltip=folium.features.GeoJsonTooltip(
                fields=hover_config[1],
                aliases=hover_config[0],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
            )
        )
        m.add_child(hover_layer)
        m.keep_in_front(hover_layer)


    folium.LayerControl().add_to(m)
    m.fit_bounds(m.get_bounds())

    if not default_column_exist:
        df_input = df_input.drop('vng_id', axis=1, inplace=True)
    
    return m
    
