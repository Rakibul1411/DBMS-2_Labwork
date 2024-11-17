SELECT Latitude, Longitude, Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY CUBE (Latitude, Longitude, Altitude);
