SELECT NULL AS continent, NULL AS country, city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY city 

UNION ALL

SELECT NULL AS continent, country, NULL AS city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY country

UNION ALL

SELECT continent, NULL AS country, NULL AS city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY continent

UNION ALL

SELECT NULL AS continent, country, city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY country, city

UNION ALL

SELECT continent, NULL AS country, city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY continent, city 

UNION ALL

SELECT continent, country, NULL AS city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY continent, country 

UNION ALL


SELECT continent, country, city, SUM(units_sold) AS units_sold
FROM product_sales
GROUP BY continent, country, city

UNION ALL

SELECT NULL AS continent, NULL AS country, NULL AS city, SUM(units_sold) AS units_sold
FROM product_sales
ORDER BY continent;
