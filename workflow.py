from datetime import datetime as dt
import json
import os
import time
import nibabel as nib
import numpy as np
import pandas as pd
import argparse
import csv

'''ROI Labels from FreeSurferColorLUT.txt'''
roi_labels = {
    "Thalamus": [10, 49],  # Combines left (10) and right (49)
    "Left-Hippocampus": 17,
    "Right-Hippocampus": 53,
    "Left-Amygdala": 18,
    "Right-Amygdala": 54,
    "Left-Caudate Nucleus": 11,
    "Right-Caudate Nucleus": 50,
    "Left-Putamen": 12,
    "Right-Putamen": 51,
    "Left-Globus Pallidus": 13,
    "Right-Globus Pallidus": 52,
    "Left-Subthalamic Nucleus": 26,
    "Right-Subthalamic Nucleus": 58,
    "Left-Substancia-Nigra": 27,
    "Right-Substancia-Nigra": 59,
    "Left-Red Nucleus": 28,
    "Right-Red Nucleus": 60,
    "Pons": 15,
    "Spinal Cord": 126,
    "Left-Cerebellum": [8, 7],  # Combines white matter (8) and cortex (7)
    "Right-Cerebellum": [47, 46],  # Combines white matter (47) and cortex (46)
    "Motor Cortex": [1022, 2022],  # Use labels from aparc+aseg.mgz
    # Superior Frontal Gyrus (1028) + Middle Frontal Gyrus (1027) + Inferior Frontal Gyrus (1019, 1020, 1018) + Orbital Frontal Cortex (1012, 1014)
    "Left-Prefrontal Cortex": [1028, 1027, 1019, 1020, 1018, 1012, 1014],
    # Superior Frontal Gyrus (2028) + Middle Frontal Gyrus (2027) + Inferior Frontal Gyrus (2019, 2020, 2018) + Orbital Frontal Cortex (2012, 2014)
    "Right-Prefrontal Cortex": [2028, 2027, 2019, 2020, 2018, 2012, 2014],
    "CSF": 24
}

# Define the columns of interest for each ROI
columns_of_interest = ['StudyID', 'Age']  # Initial

for roi in roi_labels.keys():
    columns_of_interest.extend([  # Extended
        f"{roi}_Volume_mm3",
        f"{roi}_ROI_Volume_mm3",
        f"{roi}_Centroid_X",
        f"{roi}_Centroid_Y",
        f"{roi}_Centroid_Z"
    ])


def run_recon_all(input_mri, study_id, studies_output):
    '''Step 1: Run Freesurfer's recon-all'''
    cmd1 = f"recon-all -i {input_mri} -s {study_id} -all -sd {studies_output}"
    print(f"[..] Running command: {cmd1}")
    os.system(cmd1)


def extract_volumes(study_id, studies_output):
    '''Step 2: Extract volumes of segmented structures using fs_stat_2_pd'''
    volumes = {}
    asegstats_file_names = ['aseg.stats']
    for asegstats_file_name in asegstats_file_names:
        asegstats_file = os.path.join(
            studies_output, study_id, 'stats', asegstats_file_name)
        print(f"[..] Looking for aseg.stats file at: {asegstats_file}")

        if not os.path.exists(asegstats_file):
            print(f"[!!] File not found: {asegstats_file}")
            return {}

        aseg_data = fs_stat_2_pd(asegstats_file)
        for index, row in aseg_data.iterrows():
            structure = row['StructName']
            volume = float(row['Volume_mm3'])
            volumes[structure] = volume
    return volumes


def extract_coordinates(aseg, structure_label):
    '''Step 3: Extract coordinates of structures and convert to RAS'''
    affine = aseg.affine
    seg_data = aseg.get_fdata()
    coords = np.argwhere(seg_data == structure_label)
    ras_coords = nib.affines.apply_affine(affine, coords)
    return ras_coords


def extract_roi_info(study_id, roi_labels, studies_output):
    '''Step 4: Extract information for all ROIs and compute centroids'''
    roi_info = {}
    aseg_file = os.path.join(studies_output, study_id, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_file)
    voxel_volume = np.prod(aseg.header.get_zooms())

    if not os.path.exists(aseg_file):
        print(f"[!!] File not found: {aseg_file}")
        return np.array([])

    for roi, label in roi_labels.items():
        if label is None:
            print(f"[!!] No label defined for {roi}")
            continue

        print(f"[..] Extracting data for {roi} (label: {label})")
        if isinstance(label, list):
            coords = []
            for lbl in label:
                coords.extend(extract_coordinates(aseg, lbl))
            coords = np.array(coords)
        else:
            coords = extract_coordinates(aseg, label)

        volume = len(coords) * voxel_volume if len(coords) > 0 else 0
        centroid = np.mean(coords, axis=0) if len(coords) > 0 else None
        roi_info[roi] = {
            'coordinates': coords,
            'volume': volume,
            'centroid': centroid
        }
    return roi_info


def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    # Calculate dot product
    dot_product = np.sum(v1 * v2, axis=1)

    # Calculate magnitudes
    magnitude_v1 = np.linalg.norm(v1, axis=1)
    magnitude_v2 = np.linalg.norm(v2, axis=1)

    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude_v1 * magnitude_v2)

    return cosine_sim


def cosine_distance(v1, v2):
    return 1-cosine_similarity(v1, v2)


def minkowski_distance(v1, v2, p=2):
    """Calculate the Minkowski distance between two vectors."""
    return np.sum(np.abs(v1 - v2) ** p, axis=1) ** (1/p)


def parse_csv(file_path, sep=';'):
    """Reads a CSV file and returns a DataFrame."""
    return pd.read_csv(file_path, sep=sep)


def initialize_dataframe(columns):
    """Initializes a DataFrame with the specified columns filled with NaN and correct dtypes."""
    dtype_dict = {col: float for col in columns if col not in [
        'StudyID', 'Age']}
    dtype_dict.update({'StudyID': str, 'Age': int})
    return pd.DataFrame(columns=columns).astype(dtype_dict)


def assemble_results(csv_files, input_path):
    """Assembles results from multiple CSV files into a single DataFrame."""
    # Initialize an empty DataFrame with the specified columns filled with NaN
    result_df = initialize_dataframe(columns_of_interest)

    for file in csv_files:
        file_path = os.path.join(input_path, file)
        df = parse_csv(file_path)

        for _, row in df.iterrows():
            if 'ROI' not in row:
                print(f'Empty row:')
                print(row)
                continue
            roi = row['ROI']
            if roi in roi_labels.keys():
                study_id = row['StudyID']
                age = row['Age']

                # Check if this SubjectID and StudyID combination already exists in the result DataFrame
                if ((result_df['StudyID'] == study_id)).any():
                    # Update existing row
                    idx = (result_df['StudyID'] == study_id)
                    result_df.loc[idx, f"{roi}_Volume_mm3"] = row['Volume_mm3']
                    result_df.loc[idx,
                                  f"{roi}_ROI_Volume_mm3"] = row['ROI_Volume_mm3']
                    result_df.loc[idx,
                                  f"{roi}_Centroid_X"] = row['ROI_Centroid_X']
                    result_df.loc[idx,
                                  f"{roi}_Centroid_Y"] = row['ROI_Centroid_Y']
                    result_df.loc[idx,
                                  f"{roi}_Centroid_Z"] = row['ROI_Centroid_Z']
                else:
                    # Create new row filled with NaN
                    new_row = pd.Series(index=result_df.columns, dtype=object)
                    new_row['StudyID'] = study_id
                    new_row['Age'] = age
                    new_row[f"{roi}_Volume_mm3"] = row['Volume_mm3']
                    new_row[f"{roi}_ROI_Volume_mm3"] = row['ROI_Volume_mm3']
                    new_row[f"{roi}_Centroid_X"] = row['ROI_Centroid_X']
                    new_row[f"{roi}_Centroid_Y"] = row['ROI_Centroid_Y']
                    new_row[f"{roi}_Centroid_Z"] = row['ROI_Centroid_Z']
                    result_df = pd.concat(
                        [result_df, pd.DataFrame([new_row])], ignore_index=True)

    return result_df


def aggregate_results(df):
    """Aggregates the results by selecting one volume and calculating distances from the Thalamus."""
    # Initialize columns for final volumes and distances
    aggregated_data = df[['StudyID', 'Age']].copy()

    for roi in roi_labels.keys():
        # Select the appropriate volume
        volume_column = f"{roi}_Volume_mm3"
        roi_volume_column = f"{roi}_ROI_Volume_mm3"
        aggregated_data[roi + '_Volume'] = np.where(
            df[volume_column].notna(), df[volume_column], df[roi_volume_column])
        aggregated_data[roi + '_THNorm_Volume'] = np.where(
            df[volume_column].notna(), df[volume_column], df[roi_volume_column])

        # Calculate distances from Thalamus
        if roi != "Thalamus":
            v1 = df[[roi + '_Centroid_X', roi +
                     '_Centroid_Y', roi + '_Centroid_Z']].values
            v2 = df[['Thalamus_Centroid_X', 'Thalamus_Centroid_Y',
                     'Thalamus_Centroid_Z']].values

            # Assign all columns at once
            aggregated_data = aggregated_data.assign(
                **{
                    roi + '_THNorm_Minkowski2': minkowski_distance(v1, v2, 2),
                    roi + '_THNorm_Cosine': cosine_distance(v1, v2),
                }
            )

    # Remove columns with all NaN values
    aggregated_data = aggregated_data.dropna(axis=1, how='all')

    return aggregated_data


def normalize_volumes(df):
    """Normalize ROI volumes by dividing by Thalamus volume."""
    for roi in roi_labels.keys():
        if roi != "Thalamus":
            df = df.drop(columns=[f"{roi}_Volume"])
            df[f"{roi}_THNorm_Volume"] = df[f"{roi}_THNorm_Volume"] / \
                df["Thalamus_THNorm_Volume"]

    df = df.drop(columns=[f"Thalamus_THNorm_Volume"])
    return df


def fs_stat_2_pd(input_file):
    '''Utility function to convert stats file to DataFrame'''
    with open(input_file, 'rt') as f:
        reader = csv.reader(f, delimiter=' ')
        csv_list = []
        for l in reader:
            # Join and split to handle multiple spaces and filter out comment lines
            cleaned_line = ' '.join(l).split()
            if len(cleaned_line) > 1 and not cleaned_line[0].startswith('#'):
                csv_list.append(cleaned_line)

    # Extract the header separately
    header = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName',
              'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']

    # Create DataFrame
    df = pd.DataFrame(csv_list, columns=header)

    # Convert appropriate columns to numeric
    numeric_columns = ['Index', 'SegId', 'NVoxels', 'Volume_mm3',
                       'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    return df


def get_structs_csv_files(path):
    """Returns a list of CSV files in the given directory."""
    return [f for f in os.listdir(path) if f.endswith('_structs.csv')]


def save_results(result_df, output_path):
    """Saves the assembled DataFrame to a CSV file."""
    result_df.to_csv(output_path, index=False, sep=';')


def save_coordinates_to_html(roi_info, output_file):
    '''Save the coordinates to an HTML file with Three.js visualization'''
    def serialize_coordinates(coords):
        return coords.tolist() if isinstance(coords, np.ndarray) else coords

    serialized_roi_info = {
        roi: {
            'coordinates': serialize_coordinates(info['coordinates']),
            'volume': info['volume'],
            'centroid': serialize_coordinates(info['centroid'])
        }
        for roi, info in roi_info.items()
    }

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D ROI Visualization</title>
    </head>
    <body style="margin: 0; padding: 0">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            const roiData = {roi_data};
            const colors = ["#caa331", "#6f71d9", "#96b642", "#563688", "#44cc7c", "#bc51a8", "#59ac4c", "#c584d5", "#396d22", "#da6295", "#48b989", "#e1556e", "#33d4d1", "#d85a49", "#5e87d3", "#c37127", "#892b60", "#76b870", "#ac4258", "#a2b864", "#ad4248", "#ccad52", "#984126", "#7a7020", "#ce9157"]
            
            function init() {{
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer();
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                let iRoi = 0;
                const sections = ['coordinates', 'centroid']
                for (const roi in roiData) {{
                    sections.forEach((roiSection) => {{
                        let coords = roiData[roi][roiSection];
                        if (roiSection === 'centroid') {{
                            coords = [roiData[roi][roiSection]];
                        }}
                        if (coords.filter(Boolean).length === 0) {{
                            return;
                        }}
                        const geometry = new THREE.BufferGeometry();
                        const points = [];
                        for (let i = 0; i < coords.length; i++) {{
                            points.push(coords[i][0], coords[i][1], coords[i][2]);
                        }}
                        const vertices = new Float32Array(points);
                        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                        let material = new THREE.PointsMaterial({{ color: colors[iRoi], size: 0.1, }});
                        if (roiSection === 'centroid') {{
                            material = new THREE.PointsMaterial({{ color: colors[iRoi], size: 5, }});
                        }}
                        const pointCloud = new THREE.Points(geometry, material);
                        scene.add(pointCloud);
                    }})
                    iRoi++;
                }};

                function onDocumentMouseWheel(event) {{
                    camera.fov += event.deltaY * 0.05;
                    camera.fov = Math.max(10, Math.min(100, camera.fov));
                    camera.updateProjectionMatrix();
                }}
                document.addEventListener('wheel', onDocumentMouseWheel, false);

                function animate() {{
                    const r = Date.now() * 0.0005;
                    camera.position.x = 180 * Math.cos(r);
                    camera.position.z = 180 * Math.sin(r);
                    camera.lookAt(scene.position);
                    requestAnimationFrame(animate);
                    renderer.render(scene, camera);
                }}

                scene.fog = new THREE.FogExp2( 0x000000, 0.001 );
                camera.position.z = 40;
                camera.position.y = 0;
                animate();
            }}

            init();
        </script>
    </body>
    </html>
    """

    roi_data_json = json.dumps(serialized_roi_info)
    html_content = html_template.format(roi_data=roi_data_json)

    with open(output_file, 'w') as f:
        f.write(html_content)


def main():
    '''Pipeline execution'''
    parser = argparse.ArgumentParser(description='This is input args')
    parser.add_argument('--mri', required=True,
                        help=' ++ please provide path to MRI (*.nii) for processing')
    parser.add_argument('--id', required=True,
                        help=' ++ please provide subject id')
    parser.add_argument('--age', required=True,
                        help=' ++ please provide age of subject', type=int)
    parser.add_argument('--stud_out', required=False,
                        help=' ++ optional studies output path (default = studies_output)', default=f"studies_output")
    parser.add_argument('--csv_out', required=False,
                        help=' ++ optional csv output path (default = processed_data)', default=f"processed_data")

    args = parser.parse_args()
    path_mri = args.mri
    study_id = args.id
    age = args.age
    studies_output = args.stud_out
    csv_out = args.csv_out

    if not os.path.exists(studies_output):
        os.makedirs(studies_output)

    if not os.path.exists(csv_out):
        os.makedirs(csv_out)

    # Print the results to verify
    print("[ok] Input MRI:", path_mri)
    print("[ok] Study ID:", study_id)
    print("[ok] Output directory:", studies_output)

    # Run
    run_recon_all(path_mri, study_id, studies_output)

    # Wait for recon-all to complete
    recon_all_output_dir = os.path.join(studies_output, study_id)
    while not os.path.exists(os.path.join(recon_all_output_dir, 'scripts', 'recon-all.done')):
        print("[..] Waiting for recon-all to complete...")
        time.sleep(60)

    volumes = extract_volumes(study_id, studies_output)
    roi_info = extract_roi_info(study_id, roi_labels, studies_output)

    # Write the results to a CSV file
    now_date = dt.now().strftime("%Y%m%d%H%M%S")
    output_csv = os.path.join(csv_out, f'{now_date}_{study_id}_structs.csv')
    fieldnames = ['StudyID', 'Age', 'ROI', 'Volume_mm3', 'ROI_Centroid_X',
                  'ROI_Centroid_Y', 'ROI_Centroid_Z', 'ROI_Volume_mm3']

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        # Combine volumes from aseg.stats and ROI information
        for roi in set(volumes.keys()).union(roi_info.keys()):
            volume = volumes.get(roi, None)
            info = roi_info.get(roi, {})
            centroid = info.get('centroid', None)
            roi_volume = info.get('volume', None)
            centroid_x, centroid_y, centroid_z = centroid if centroid is not None else (
                None, None, None)

            writer.writerow({
                'StudyID': study_id,
                'Age': age,
                'ROI': roi,
                'Volume_mm3': volume,
                'ROI_Centroid_X': centroid_x,
                'ROI_Centroid_Y': centroid_y,
                'ROI_Centroid_Z': centroid_z,
                'ROI_Volume_mm3': roi_volume
            })

    print(f"[ok] Raw structs written to {output_csv}")

    # Get list of CSV files in the directory
    csv_files = get_structs_csv_files(csv_out)

    # Assemble the results from the CSV files
    result_df = assemble_results(csv_files, csv_out)

    # Aggregate the results
    aggregated_df = aggregate_results(result_df)

    # Normalize volumes
    normalized_df = normalize_volumes(aggregated_df)

    # Save
    output_results_csv = os.path.join(csv_out, f'{now_date}_norm.csv')
    save_results(normalized_df, output_results_csv)

    # Coords
    output_results_points = os.path.join(
        csv_out, f'{now_date}_{study_id}_s3d.html')
    save_coordinates_to_html(roi_info, output_results_points)
    print(f"[ok] 3d visualization written to {output_results_points}")

    # Finish and inform
    print(f"[ok] Normalized results written to {output_results_csv}")
    print("[ok] Processing is finished.")


if __name__ == '__main__':
    main()
