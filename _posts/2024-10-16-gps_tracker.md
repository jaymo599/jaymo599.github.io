```python
from flask import Flask, jsonify, request

app = Flask(__name__)

# GPS 데이터 저장소 (기본값: 서울 좌표)
gps_data = {"lat": 37.5665, "lon": 126.9780}

# 스마트폰에서 GPS 데이터를 받는 엔드포인트
@app.route('/update_gps', methods=['POST'])
def update_gps():
    global gps_data
    data = request.get_json()
    gps_data['lat'] = data.get('lat', gps_data['lat'])
    gps_data['lon'] = data.get('lon', gps_data['lon'])
    return jsonify({"status": "GPS data updated successfully!"})

# HTML과 지도를 렌더링하고, GPS 데이터를 실시간으로 반영
@app.route('/')
def map():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPS Tracker</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script>
            // 페이지 로드 시 스마트폰에서 GPS 데이터를 수집하는 함수
            function getGPSData() {
                if (navigator.geolocation) {
                    navigator.geolocation.watchPosition(function(position) {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;

                        // 서버로 스마트폰의 GPS 데이터를 전송
                        fetch('/update_gps', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({lat: lat, lon: lon}),
                        })
                        .then(response => response.json())
                        .then(data => console.log('GPS Data Sent:', data))
                        .catch(error => console.error('Error:', error));
                    }, function(error) {
                        console.error('Error occurred while fetching GPS data:', error);
                    }, {
                        enableHighAccuracy: true,
                        timeout: 5000,
                        maximumAge: 0
                    });
                } else {
                    alert("Geolocation is not supported by this browser.");
                }
            }

            // 지도 초기화
            function initMap() {
                var map = L.map('map').setView([37.5665, 126.9780], 13);  // 기본 서울 좌표
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);

                var marker = null;  // 마커를 저장할 변수
                var follow = true;  // 지도를 GPS 위치로 계속 추적할지 여부

                // 지도 추적을 활성화/비활성화하는 버튼 추가
                var followButton = L.control({position: 'topright'});
                followButton.onAdd = function () {
                    var div = L.DomUtil.create('div', 'follow-button');
                    div.innerHTML = '<button onclick="toggleFollow()">GPS 추적</button>';
                    div.style.backgroundColor = 'white';
                    div.style.padding = '5px';
                    return div;
                };
                followButton.addTo(map);

                // GPS 추적을 활성화/비활성화하는 함수
                function toggleFollow() {
                    follow = !follow;
                    console.log('GPS 추적: ' + (follow ? '활성화' : '비활성화'));
                }

                // 서버에서 GPS 데이터를 주기적으로 받아서 지도를 업데이트
                function updateMap() {
                    fetch('/gps')
                    .then(response => response.json())
                    .then(data => {
                        if (marker) {
                            map.removeLayer(marker);  // 이전 마커 제거
                        }
                        marker = L.marker([data.lat, data.lon]).addTo(map);  // 새로운 마커 추가

                        if (follow) {
                            map.setView([data.lat, data.lon], 13);  // 지도를 GPS 위치로 이동
                        }
                    });
                }

                // 5초마다 GPS 데이터를 업데이트
                setInterval(updateMap, 5000);
            }

            // 페이지가 로드되면 지도 초기화 및 GPS 데이터 수집 시작
            window.onload = function() {
                initMap();
                getGPSData();  // GPS 데이터 수집 시작
            };
        </script>
    </head>
    <body>
        <h1>실시간 GPS 추적</h1>
        <div id="map" style="width: 100%; height: 400px;"></div>
    </body>
    </html>
    """

# GPS 데이터를 제공하는 엔드포인트
@app.route('/gps')
def gps_endpoint():
    return jsonify(gps_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```
