import streamlit as st


def review_detections_ui(debug_image, crops):
    """
    Kullanıcıya detect edilen objeleri gösterir.
    Seçilen crop listesini döner.
    """

    st.subheader("Detected objects")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(debug_image, use_container_width=True)

    with col2:
        st.markdown("### Select objects to keep")

        selected_indices = []

        for i, crop in enumerate(crops):
            st.image(crop["image"], use_container_width=True)

            keep = st.checkbox(
                f'{crop["label"]} ({crop["score"]:.2f})',
                value=True,
                key=f"keep_{i}"
            )

            if keep:
                selected_indices.append(i)

    st.divider()

    if st.button("Next"):
        selected_crops = [crops[i] for i in selected_indices]
        return selected_crops

    return None
