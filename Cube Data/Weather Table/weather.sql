CREATE TABLE Weather (
    Time TIMESTAMP,          
    Latitude VARCHAR(15),
    Longitude VARCHAR(15),
    Altitude INT,
    Temperature DECIMAL(5,2),
    Pressure DECIMAL(6,2)
);


INSERT INTO Weather (Time, Latitude, Longitude, Altitude, Temperature, Pressure)
VALUES
    ('1996-06-01 15:00:00', '37:58:33N', '122:45:28W', 102, 21, 1009),
    ('1996-06-07 15:00:00', '34:16:18N', '27:05:55W', 10, 23, 1024);
