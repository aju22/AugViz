import streamlit as st
import albumentations as A
from PIL import Image
import numpy as np
import math
from typing import Callable, get_args
from utils import get_transforms_map, apply_augmentations, is_range_param, get_params_info, infer_type_from_annotation

# Create input for parameter based on type
def create_input_for_param(param_name, param_info, key_prefix):
    
    param_type = infer_type_from_annotation(param_info['annotation'])
    default_value = param_info['default']
    required = param_info['required']

    BUFFER_INT = 5
    BUFFER_FlOAT = 0.5

    
    if required:
        if param_type == 'int':
            return int(st.text_input(f"{param_name} (required int)", key=f"{key_prefix}_{param_name}", value='0'))
        elif param_type == 'float':
            return float(st.text_input(f"{param_name} (required float)", key=f"{key_prefix}_{param_name}", value='0'))
        elif param_type == 'str':
            return st.text_input(f"{param_name} (required str)", key=f"{key_prefix}_{param_name}")
        elif param_type == 'tuple':
            values_tuple = st.text_input(f"{param_name} (required tuple)", key=f"{key_prefix}_{param_name}", value='(0, 0)')
            values_tuple = values_tuple[1:-1]  # Remove parentheses
            try:
                if ',' in values_tuple:
                    values = values_tuple.split(',')
                    if 'int' in get_args(param_info['annotation']):
                        return (int(val) for val in values)
                    else:
                        return (float(val) for val in values)
            except:
                st.error(f"Unable to parse input. Please enter a valid tuple of values(int | float).")

    
    else:
        
        if isinstance(default_value, bool):
            return st.checkbox(param_name, default_value if default_value is not None else False, key=f"{key_prefix}_{param_name}")
        
        elif isinstance(default_value, int):
            
            min_val = st.session_state.get(f"{key_prefix}_{param_name}_min", default_value - BUFFER_INT)
            max_val = st.session_state.get(f"{key_prefix}_{param_name}_max", default_value + BUFFER_INT)

            # Display slider
            slider_placeholder = st.empty()

            # Create columns for buttons
            col1, _, col2 = st.columns([0.1, 1, 0.1])
            with col1:
                slider_min = st.button("➖", key=f"minus_{key_prefix}_{param_name}")
            with col2:
                slider_max = st.button("➕", key=f"plus_{key_prefix}_{param_name}")

            # Adjust values based on button clicks
            if slider_min:
                min_val -= BUFFER_INT
                st.session_state[f"{key_prefix}_{param_name}_min"] = min_val  # Update session state
            if slider_max:
                max_val += BUFFER_INT
                st.session_state[f"{key_prefix}_{param_name}_max"] = max_val  # Update session state

            value = slider_placeholder.slider(param_name, 
                                              min_value=min_val, 
                                              max_value=max_val, 
                                              value=default_value, 
                                              step=1, key=f"{key_prefix}_{param_name}")

            return value

        elif isinstance(default_value, float):
            min_val = st.session_state.get(f"{key_prefix}_{param_name}_min", default_value - BUFFER_FlOAT)
            max_val = st.session_state.get(f"{key_prefix}_{param_name}_max", default_value + BUFFER_FlOAT)

            # Display slider
            slider_placeholder = st.empty()

            # Create columns for buttons
            col1, _, col2 = st.columns([0.1, 1, 0.1])
            with col1:
                slider_min = st.button("➖", key=f"minus_{key_prefix}_{param_name}")
            with col2:
                slider_max = st.button("➕", key=f"plus_{key_prefix}_{param_name}")

            # Adjust values based on button clicks
            if slider_min:
                min_val -= BUFFER_FlOAT
                st.session_state[f"{key_prefix}_{param_name}_min"] = min_val  # Update session state
            if slider_max:
                max_val += BUFFER_FlOAT
                st.session_state[f"{key_prefix}_{param_name}_max"] = max_val  # Update session state

            value = slider_placeholder.slider(param_name, 
                                              min_value=float(min_val), 
                                              max_value=float(max_val), 
                                              value=default_value, 
                                              step=0.01, 
                                              key=f"{key_prefix}_{param_name}")

            return value

        elif isinstance(default_value, str):
            return default_value  # Don't allow user input for default str params
        
        elif is_range_param(default_value):
            min_val, max_val = default_value
            if isinstance(min_val, int) and isinstance(max_val, int):
                new_min_val = st.session_state.get(f"{key_prefix}_{param_name}_min", min_val - BUFFER_INT)
                new_max_val = st.session_state.get(f"{key_prefix}_{param_name}_max", max(abs(min_val), max_val) + BUFFER_INT) 

                slider_placeholder = st.empty()

                col1, _, col2 = st.columns([0.1, 1, 0.1])
                
                with col1:
                    slider_min = st.button("➖", key=f"minus_{key_prefix}_{param_name}")
                with col2:
                    slider_max = st.button("➕", key=f"plus_{key_prefix}_{param_name}")
                 
                if slider_min:
                    new_min_val -= BUFFER_INT
                    st.session_state[f"{key_prefix}_{param_name}_min"] = new_min_val  # Update session state
                if slider_max:
                    new_max_val += BUFFER_INT
                    st.session_state[f"{key_prefix}_{param_name}_max"] = new_max_val  # Update session state

                value = slider_placeholder.slider(param_name, 
                                  min_value=new_min_val, 
                                  max_value=new_max_val, 
                                  value=(min_val, max_val), 
                                  step=1, 
                                  key=f"{key_prefix}_{param_name}")

                return value
            
            else:

                new_min_val = st.session_state.get(f"{key_prefix}_{param_name}_min", min_val - BUFFER_FlOAT)
                new_max_val = st.session_state.get(f"{key_prefix}_{param_name}_max", max(abs(min_val), max_val) + BUFFER_FlOAT)

                # Display slider
                slider_placeholder = st.empty()
                
                # Create columns for buttons
                col1, _, col2 = st.columns([0.1, 1, 0.1])
                
                with col1:
                    slider_min = st.button("➖", key=f"minus_{key_prefix}_{param_name}")
                with col2:
                    slider_max = st.button("➕", key=f"plus_{key_prefix}_{param_name}")

                # Adjust values based on button clicks
                if slider_min:
                    new_min_val -= BUFFER_FlOAT
                    st.session_state[f"{key_prefix}_{param_name}_min"] = new_min_val  # Update session state
                if slider_max:
                    new_max_val += BUFFER_FlOAT
                    st.session_state[f"{key_prefix}_{param_name}_max"] = new_max_val  # Update session state

                value = slider_placeholder.slider(
                    param_name,
                    min_value=float(new_min_val),
                    max_value=float(new_max_val),
                    value=(float(min_val), float(max_val)),
                    step=0.01,
                    key=f"{key_prefix}_{param_name}"
                )
                
                return value

        elif isinstance(default_value, tuple):
            values_tuple = st.text_input(f"{param_name}", key=f"{key_prefix}_{param_name}", value=default_value)
            values_tuple = values_tuple[1:-1]  # Remove parentheses
            try:
                if ',' in values_tuple:
                    values = values_tuple.split(',')
                    if 'int' in get_args(param_info['annotation']):
                        return (int(val) for val in values)
                    else:
                        return (float(val) for val in values)
            except:
                st.error(f"Unable to parse input. Please enter a valid tuple of values(int | float).")
        
        else:
            # For any other types, display but don't allow editing
            st.text(f"{param_name}: {default_value}")
            return default_value

def main():
    st.set_page_config(layout="wide")
    st.title("Welcome to AugViz!")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    error_placeholder = st.empty()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display original image with adjustable width
        st.header("Original Image")

        _, col, _ = st.columns([0.3, 1, 0.3])
        with col:
            st.image(image, width=400)

        
        
        st.sidebar.header("Augmentation Settings")
        
        transforms_map = get_transforms_map()
        transforms_keys = list(sorted(transforms_map.keys()))
        
        selected_augs = st.sidebar.multiselect("Select Augmentations", transforms_keys)
        
        augmentations = []
        
        if selected_augs:
            tabs = st.sidebar.tabs(selected_augs)
        
        for i, aug_name in enumerate(selected_augs):
            with tabs[i]:
                aug_class = transforms_map[aug_name]
                params_info = get_params_info(aug_class)
                
                with st.expander("Show Documentation"):
                    st.markdown(aug_class.__doc__)
                
                aug_args = {}
                
                for param_name, param_info in params_info.items():
                    if param_name in ['always_apply']:
                        continue
                    
                    value = create_input_for_param(param_name, param_info, aug_name)
                    if value is not None:  # Only add if a value was returned
                        aug_args[param_name] = value

                try:
                    aug = aug_class(**aug_args)
                except Exception as e:
                    error_placeholder.error(f"Error creating augmentation: {e}")
                    break
                
                augmentations.append(aug)
                
    
        
        
        # Apply and display augmented images
        if augmentations:
            st.header("Augmented Images")
            num_samples = st.number_input("Number of sample images to display", min_value=1, max_value=50, value=5)
            
            # Calculate number of columns and rows
            num_cols = min(5, num_samples)
            num_rows = math.ceil(num_samples / num_cols)
            
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col in range(num_cols):
                    index = row * num_cols + col
                    if index < num_samples:
                        result = apply_augmentations(image_np, augmentations)
                        if isinstance(result, tuple):  # Error occurred
                            st.error(f"Error applying augmentation: {result[1]}")
                            break
                        else:
                            cols[col].image(result, use_column_width=True)
        else:
            st.session_state.clear()  # Clear session state if no augmentations are selected

    st.sidebar.markdown("---")
    st.sidebar.markdown("# About 💡")
    st.sidebar.markdown(
        "This tool allows you to directly visualize the effects of various augmentations from the [Albumentations](https://albumentations.ai/) library on your images. "
    )
    st.sidebar.markdown(
        "This tool is a work in progress.  "
    )

    st.sidebar.markdown("If you encounter any issues or have suggestions for improvements, please feel free to reach out to me.")
    st.sidebar.markdown("Drop an ✉️ at arjun.krishna@airamatrix.com")
    st.sidebar.markdown("Made by Arjun")
    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()