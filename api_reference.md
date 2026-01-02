# API Reference Documentation

## Solar Emission Projection API v1.0

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, no authentication is required for development. Production deployments should implement API keys or OAuth2.

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check API health and resource availability.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 3,
  "data_available": true
}
```

---

### 2. Get Scenarios

**GET** `/api/v1/scenarios`

Retrieve available IEA scenarios and their parameters.

**Response:**
```json
[
  {
    "scenario": "NZE",
    "name": "Net Zero Emissions by 2050",
    "description": "1.5°C pathway with aggressive renewables deployment",
    "solar_growth_rate": 0.15,
    "carbon_price_2030": 130
  },
  ...
]
```

---

### 3. Get Regions

**GET** `/api/v1/regions`

Get list of available regions.

**Response:**
```json
["Asia", "Europe", "North America", "South America", "Oceania", "Other"]
```

---

### 4. Get Projections

**GET** `/api/v1/projections`

Get emission projections for a specific scenario and year.

**Parameters:**
- `scenario` (required): Scenario name (NZE, APS, or STEPS)
- `year` (required): Projection year (2025-2050)
- `region` (optional): Filter by region

**Example Request:**
```bash
GET /api/v1/projections?scenario=NZE&year=2030&region=Asia
```

**Response:**
```json
[
  {
    "scenario": "NZE",
    "year": 2030,
    "region": "Asia",
    "capacity_mw": 1250000.50,
    "generation_mwh": 2750000000.00,
    "emissions_avoided_tco2e": 1200000000.00
  }
]
```

---

### 5. Compare Scenarios

**GET** `/api/v1/projections/compare`

Compare all scenarios for a specific year and region.

**Parameters:**
- `year` (required): Projection year (2025-2050)
- `region` (optional): Filter by region

**Example Request:**
```bash
GET /api/v1/projections/compare?year=2030&region=Asia
```

**Response:**
```json
[
  {
    "region": "Asia",
    "NZE": 1200000000.00,
    "APS": 950000000.00,
    "STEPS": 750000000.00
  }
]
```

---

### 6. Get Risk Metrics

**GET** `/api/v1/risk`

Get transition risk metrics for a specific year and region.

**Parameters:**
- `year` (required): Projection year (2025-2050)
- `region` (optional): Filter by region

**Example Request:**
```bash
GET /api/v1/risk?year=2030&region=Asia
```

**Response:**
```json
[
  {
    "year": 2030,
    "region": "Asia",
    "transition_risk_score": 0.456,
    "policy_risk_score": 0.389,
    "stranded_asset_exposure": 125000000.00,
    "nze_aps_gap": 250000000.00,
    "aps_steps_gap": 200000000.00
  }
]
```

---

### 7. Get Risk Summary

**GET** `/api/v1/risk/summary`

Get aggregated risk summary across all regions.

**Parameters:**
- `year` (optional, default: 2030): Projection year

**Example Request:**
```bash
GET /api/v1/risk/summary?year=2030
```

**Response:**
```json
{
  "year": 2030,
  "global_metrics": {
    "avg_transition_risk": 0.423,
    "avg_policy_risk": 0.365,
    "total_stranded_asset_exposure": 450000000.00,
    "max_nze_steps_gap": 500000000.00
  },
  "high_risk_regions": [
    {
      "region": "Asia",
      "transition_risk_score": 0.567
    },
    ...
  ]
}
```

---

### 8. Get Timeline

**GET** `/api/v1/timeline`

Get emissions timeline for a scenario and region.

**Parameters:**
- `scenario` (optional, default: NZE): Scenario name
- `region` (optional, default: Asia): Region name

**Example Request:**
```bash
GET /api/v1/timeline?scenario=NZE&region=Asia
```

**Response:**
```json
{
  "scenario": "NZE",
  "region": "Asia",
  "timeline": [
    {
      "year": 2025,
      "emissions_avoided_tco2e": 800000000.00
    },
    {
      "year": 2030,
      "emissions_avoided_tco2e": 1200000000.00
    },
    ...
  ]
}
```

---

## Error Responses

### 400 Bad Request
Invalid parameters provided.
```json
{
  "detail": "Invalid scenario. Use NZE, APS, or STEPS"
}
```

### 404 Not Found
No data found for the specified parameters.
```json
{
  "detail": "No projections found for specified parameters"
}
```

### 503 Service Unavailable
Required data or models not loaded.
```json
{
  "detail": "Projection data not available"
}
```

---

## Usage Examples

### Python
```python
import requests

# Get projections
response = requests.get(
    "http://localhost:8000/api/v1/projections",
    params={
        "scenario": "NZE",
        "year": 2030,
        "region": "Asia"
    }
)
data = response.json()
print(data)
```

### curl
```bash
# Health check
curl http://localhost:8000/health

# Get scenarios
curl http://localhost:8000/api/v1/scenarios

# Get projections
curl "http://localhost:8000/api/v1/projections?scenario=NZE&year=2030&region=Asia"

# Compare scenarios
curl "http://localhost:8000/api/v1/projections/compare?year=2030"

# Get risk metrics
curl "http://localhost:8000/api/v1/risk?year=2030&region=Asia"
```

### JavaScript (fetch)
```javascript
// Get projections
fetch('http://localhost:8000/api/v1/projections?scenario=NZE&year=2030')
  .then(response => response.json())
  .then(data => console.log(data));

// Compare scenarios
fetch('http://localhost:8000/api/v1/projections/compare?year=2030')
  .then(response => response.json())
  .then(data => console.log(data));
```

---

## Rate Limiting

Currently no rate limiting is implemented. Production deployments should implement:
- 100 requests per minute per IP
- 1000 requests per hour per API key

---

## Versioning

API versioning is handled through URL path (`/api/v1/`). Future versions will be available at `/api/v2/`, etc.

---

## Support

For issues or questions:
- GitHub Issues: [repository-url]/issues
- Email: support@example.com
- Documentation: [repository-url]/docs
