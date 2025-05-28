# Coral Bleaching Dataset Documentation

## Overview
This dataset combines multiple data sources to create a comprehensive coral bleaching monitoring system. It integrates satellite imagery, environmental parameters, and ground truth survey data to enable both predictive modeling and retrospective analysis of coral health.

## Directory Structure
```
data/
├── noaa_crw/              # NOAA Coral Reef Watch data
│   ├── metadata.json      # Data source metadata
│   └── *.nc              # NetCDF files with SST, DHW, etc.
├── reef_check/            # Ground truth survey data
│   ├── metadata.json      # Survey metadata
│   └── reef_check_surveys.csv  # Survey records
├── satellite_imagery/     # High-resolution satellite imagery
│   ├── location_000/      # Location-specific imagery
│   │   ├── metadata.json  # Location metadata
│   │   └── *.tif         # GeoTIFF satellite images
│   └── location_001/
└── environmental/         # Environmental parameters
    ├── metadata.json      # Parameters metadata
    └── environmental_*.csv  # Time series data
```

## Data Sources and Mappings

### 1. NOAA Coral Reef Watch Data
- **Source**: NOAA Coral Reef Watch 5km resolution products
- **Parameters**:
  - Sea Surface Temperature (SST)
  - Degree Heating Weeks (DHW)
  - Hotspot
  - Bleaching Alert Area
- **Temporal Resolution**: Daily
- **Spatial Resolution**: 5km
- **File Format**: NetCDF (.nc)
- **Mapping for Analysis**:
  ```python
  {
      'sst': 'temperature',
      'dhw': 'degree_heating_weeks',
      'hotspot': 'thermal_stress',
      'bleaching_alert': 'alert_level'
  }
  ```

### 2. ReefCheck Survey Data
- **Source**: ReefCheck Global Database
- **Parameters**:
  - Hard Coral Cover (%)
  - Soft Coral Cover (%)
  - Bleached Coral (%)
  - Pale Coral (%)
  - Healthy Coral (%)
  - Dead Coral (%)
  - Rubble (%)
  - Environmental Conditions
- **File Format**: CSV
- **Mapping for Analysis**:
  ```python
  {
      'hard_coral_cover': 'total_coral_cover',
      'soft_coral_cover': 'soft_coral_percentage',
      'bleached_coral': 'bleaching_severity',
      'pale_coral': 'stress_indicator',
      'healthy_coral': 'health_status',
      'dead_coral': 'mortality',
      'rubble': 'reef_degradation'
  }
  ```

### 3. Satellite Imagery
- **Sources**: 
  - Landsat 8/9
  - Sentinel-2
- **Parameters**:
  - RGB bands
  - NIR bands
  - SWIR bands
- **Resolution**: 10-30m
- **File Format**: GeoTIFF (.tif)
- **Mapping for Analysis**:
  ```python
  {
      'B02': 'blue_band',
      'B03': 'green_band',
      'B04': 'red_band',
      'B08': 'nir_band',
      'B11': 'swir_band'
  }
  ```

### 4. Environmental Parameters
- **Parameters**:
  - Sea Surface Temperature (°C)
  - pH
  - Aragonite Saturation
  - Dissolved Oxygen (mg/L)
  - Turbidity (NTU)
  - Chlorophyll-a (mg/m³)
  - Nitrates (μmol/L)
  - Phosphates (μmol/L)
  - Salinity (PSU)
  - Wave Height (m)
  - Wind Speed (m/s)
  - Solar Radiation (W/m²)
- **Temporal Resolution**: Daily
- **File Format**: CSV
- **Mapping for Analysis**:
  ```python
  {
      'sst': 'temperature',
      'ph': 'acidity',
      'aragonite_saturation': 'calcification_potential',
      'dissolved_oxygen': 'oxygen_levels',
      'turbidity': 'water_clarity',
      'chlorophyll_a': 'nutrient_availability',
      'nitrates': 'nitrogen_levels',
      'phosphates': 'phosphorus_levels',
      'salinity': 'salt_content',
      'wave_height': 'wave_energy',
      'wind_speed': 'wind_energy',
      'solar_radiation': 'light_availability'
  }
  ```

## Data Integration for Analysis

### Temporal Alignment
- All data sources are aligned to daily timestamps
- Satellite imagery is available monthly
- Survey data is available at irregular intervals
- Environmental parameters are available daily

### Spatial Alignment
- NOAA data: 5km grid
- Satellite imagery: 10-30m resolution
- Survey data: Point locations
- Environmental data: Point locations

### Feature Engineering Pipeline
```python
def prepare_features(data_dict):
    """
    Prepare features for machine learning pipeline
    
    Args:
        data_dict: Dictionary containing all data sources
        
    Returns:
        dict: Processed features ready for model training
    """
    features = {
        'temporal_features': {
            'sst_trend': calculate_trend(data_dict['noaa']['sst']),
            'dhw_accumulation': calculate_dhw(data_dict['noaa']['dhw']),
            'environmental_conditions': aggregate_environmental(data_dict['environmental'])
        },
        'spatial_features': {
            'satellite_indices': calculate_spectral_indices(data_dict['satellite']),
            'reef_health': aggregate_survey_data(data_dict['reef_check'])
        },
        'target_variables': {
            'bleaching_status': data_dict['reef_check']['bleached_coral'],
            'health_status': data_dict['reef_check']['health_status']
        }
    }
    return features
```

## Model Training Considerations

### Input Features
1. **Environmental Features**
   - SST trends and anomalies
   - Degree Heating Weeks
   - Water quality parameters
   - Weather conditions

2. **Satellite Features**
   - Spectral indices (NDVI, NDWI)
   - Texture features
   - Temporal changes

3. **Survey Features**
   - Coral cover percentages
   - Health status indicators
   - Environmental conditions

### Target Variables
1. **Bleaching Prediction**
   - Binary classification (bleached/not bleached)
   - Multi-class classification (severity levels)
   - Regression (percentage of bleached coral)

2. **Health Status Prediction**
   - Multi-class classification (healthy, stressed, bleached, dead)
   - Regression (health score)

### Evaluation Metrics
- Accuracy
- F1 Score
- Mean Squared Error
- R² Score
- Area Under ROC Curve

## Data Quality and Preprocessing

### Quality Checks
1. **Temporal Consistency**
   - Check for missing dates
   - Validate time series continuity
   - Handle irregular sampling

2. **Spatial Consistency**
   - Validate coordinate systems
   - Check for spatial gaps
   - Handle resolution mismatches

3. **Data Validation**
   - Range checks for parameters
   - Outlier detection
   - Missing value handling

### Preprocessing Steps
1. **Temporal Processing**
   - Resampling to consistent intervals
   - Trend removal
   - Seasonal decomposition

2. **Spatial Processing**
   - Coordinate transformation
   - Resolution matching
   - Spatial interpolation

3. **Feature Processing**
   - Normalization
   - Standardization
   - Feature scaling

## Future Improvements
1. **Data Collection**
   - Increase temporal resolution
   - Add more survey locations
   - Include additional environmental parameters

2. **Processing Pipeline**
   - Implement automated quality checks
   - Add real-time processing capabilities
   - Improve spatial alignment

3. **Analysis Pipeline**
   - Add deep learning models
   - Implement ensemble methods
   - Include uncertainty quantification 