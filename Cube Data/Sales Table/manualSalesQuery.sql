SELECT Model, Year, Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Model, Year, Color

UNION ALL

SELECT Model, Year, NULL AS Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Model, Year

UNION ALL

SELECT Model, NULL AS Year, Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Model, Color

UNION ALL

SELECT NULL AS Model, Year, Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Year, Color

UNION ALL

SELECT Model, NULL AS Year, NULL AS Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Model

UNION ALL

SELECT NULL AS Model, Year, NULL AS Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Year

UNION ALL

SELECT NULL AS Model, NULL AS Year, Color, SUM(Sales) AS TotalSales
FROM Sales
GROUP BY Color

UNION ALL

SELECT NULL AS Model, NULL AS Year, NULL AS Color, SUM(Sales) AS TotalSales
FROM Sales;
