SELECT Model, Year, Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY CUBE (Model, Year, Color);
