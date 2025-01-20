SELECT continent, country, city, SUM(units_sold) AS total_sold
FROM product_sales
GROUP BY CUBE(continent, country, city)
ORDER BY city;




SELECT continent, country, SUM(units_sold) AS total_sold
FROM product_sales
GROUP BY CUBE(continent);