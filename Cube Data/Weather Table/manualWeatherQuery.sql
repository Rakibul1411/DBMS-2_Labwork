SELECT Latitude, Longitude, Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Latitude, Longitude, Altitude

UNION ALL

SELECT Latitude, Longitude, NULL AS Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Latitude, Longitude

UNION ALL

SELECT Latitude, NULL AS Longitude, Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Latitude, Altitude

UNION ALL

SELECT NULL AS Latitude, Longitude, Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Longitude, Altitude

UNION ALL

SELECT Latitude, NULL AS Longitude, NULL AS Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Latitude

UNION ALL

SELECT NULL AS Latitude, Longitude, NULL AS Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Longitude

UNION ALL

SELECT NULL AS Latitude, NULL AS Longitude, Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather
GROUP BY Altitude

UNION ALL

SELECT NULL AS Latitude, NULL AS Longitude, NULL AS Altitude,
       SUM(Temperature) AS TotalTemperature,
       SUM(Pressure) AS TotalPressure
FROM Weather;

