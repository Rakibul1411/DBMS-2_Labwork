// LibraryItem interface
interface LibraryItem {
    String getDetails();
    boolean borrowItem(User user);
    String getItemId();
    boolean isRestricted();
}

// Concrete LibraryItem implementations
class Book implements LibraryItem {
    private String itemId;
    private String title;
    private String author;
    private boolean isRestricted;
    private boolean isAvailable = true;

    public Book(String itemId, String title, String author, boolean isRestricted) {
        this.itemId = itemId;
        this.title = title;
        this.author = author;
        this.isRestricted = isRestricted;
    }

    @Override
    public String getDetails() {
        return String.format("Book - ID: %s, Title: %s, Author: %s", itemId, title, author);
    }

    @Override
    public boolean borrowItem(User user) {
        if (isAvailable) {
            isAvailable = false;
            return true;
        }
        return false;
    }

    @Override
    public String getItemId() {
        return itemId;
    }

    @Override
    public boolean isRestricted() {
        return isRestricted;
    }
}

class Magazine implements LibraryItem {
    private String itemId;
    private String title;
    private String issue;
    private boolean isRestricted;
    private boolean isAvailable = true;

    public Magazine(String itemId, String title, String issue, boolean isRestricted) {
        this.itemId = itemId;
        this.title = title;
        this.issue = issue;
        this.isRestricted = isRestricted;
    }

    @Override
    public String getDetails() {
        return String.format("Magazine - ID: %s, Title: %s, Issue: %s", itemId, title, issue);
    }

    @Override
    public boolean borrowItem(User user) {
        if (isAvailable) {
            isAvailable = false;
            return true;
        }
        return false;
    }

    @Override
    public String getItemId() {
        return itemId;
    }

    @Override
    public boolean isRestricted() {
        return isRestricted;
    }
}

// Factory Pattern
class LibraryItemFactory {
    public static LibraryItem createLibraryItem(String type, String itemId, String title, 
                                              String authorOrIssue, boolean isRestricted) {
        switch (type.toLowerCase()) {
            case "book":
                return new Book(itemId, title, authorOrIssue, isRestricted);
            case "magazine":
                return new Magazine(itemId, title, authorOrIssue, isRestricted);
            default:
                throw new IllegalArgumentException("Unknown library item type: " + type);
        }
    }
}

// User class
class User {
    private String userId;
    private String name;
    private boolean hasRestrictedAccess;

    public User(String userId, String name, boolean hasRestrictedAccess) {
        this.userId = userId;
        this.name = name;
        this.hasRestrictedAccess = hasRestrictedAccess;
    }

    public boolean hasRestrictedAccess() {
        return hasRestrictedAccess;
    }

    public String getUserId() {
        return userId;
    }

    public String getName() {
        return name;
    }
}

// Library Access Interface
interface LibraryAccess {
    LibraryItem accessItem(String itemId, User user) throws Exception;
}

// Real Library Access
class RealLibraryAccess implements LibraryAccess {
    private Map<String, LibraryItem> items = new HashMap<>();

    public void addItem(LibraryItem item) {
        items.put(item.getItemId(), item);
    }

    @Override
    public LibraryItem accessItem(String itemId, User user) throws Exception {
        LibraryItem item = items.get(itemId);
        if (item == null) {
            throw new Exception("Item not found: " + itemId);
        }
        return item;
    }
}

// Proxy Pattern
class LibraryAccessProxy implements LibraryAccess {
    private RealLibraryAccess realAccess;

    public LibraryAccessProxy(RealLibraryAccess realAccess) {
        this.realAccess = realAccess;
    }

    @Override
    public LibraryItem accessItem(String itemId, User user) throws Exception {
        LibraryItem item = realAccess.accessItem(itemId, user);
        
        if (item.isRestricted() && !user.hasRestrictedAccess()) {
            throw new Exception("Access denied: This item requires special permissions");
        }
        
        // Log access
        System.out.printf("User %s accessed item %s at %s%n", 
            user.getName(), itemId, new Date());
            
        return item;
    }
}

// Singleton Pattern
class LibraryConfigManager {
    private static LibraryConfigManager instance;
    private Map<String, Object> settings;

    private LibraryConfigManager() {
        settings = new HashMap<>();
        initializeDefaultSettings();
    }

    public static synchronized LibraryConfigManager getInstance() {
        if (instance == null) {
            instance = new LibraryConfigManager();
        }
        return instance;
    }

    private void initializeDefaultSettings() {
        settings.put("lateFeePerDay", 1.0);
        settings.put("maxBorrowDays", 14);
        settings.put("openingTime", "09:00");
        settings.put("closingTime", "18:00");
    }

    public Object getSetting(String key) {
        return settings.get(key);
    }

    public void updateSetting(String key, Object value) {
        settings.put(key, value);
    }
}

// Demo Class
public class LibraryManagementSystem {
    public static void main(String[] args) {
        try {
            // Initialize the library
            RealLibraryAccess realAccess = new RealLibraryAccess();
            LibraryAccessProxy accessProxy = new LibraryAccessProxy(realAccess);

            // Create some library items using factory
            LibraryItem book1 = LibraryItemFactory.createLibraryItem("book", "B001", 
                "Design Patterns", "Gang of Four", false);
            LibraryItem book2 = LibraryItemFactory.createLibraryItem("book", "B002", 
                "Restricted Book", "John Doe", true);
            LibraryItem magazine1 = LibraryItemFactory.createLibraryItem("magazine", "M001", 
                "Tech Monthly", "Issue 45", false);

            // Add items to library
            realAccess.addItem(book1);
            realAccess.addItem(book2);
            realAccess.addItem(magazine1);

            // Create users
            User regularUser = new User("U001", "Alice", false);
            User privilegedUser = new User("U002", "Bob", true);

            // Get library configuration
            LibraryConfigManager config = LibraryConfigManager.getInstance();
            System.out.println("Library opens at: " + config.getSetting("openingTime"));

            // Test access
            System.out.println("\nTesting access for regular user:");
            try {
                LibraryItem item = accessProxy.accessItem("B001", regularUser);
                System.out.println("Access granted: " + item.getDetails());
                
                item = accessProxy.accessItem("B002", regularUser);
                System.out.println("Access granted: " + item.getDetails());
            } catch (Exception e) {
                System.out.println("Access error: " + e.getMessage());
            }

            System.out.println("\nTesting access for privileged user:");
            try {
                LibraryItem item = accessProxy.accessItem("B002", privilegedUser);
                System.out.println("Access granted: " + item.getDetails());
            } catch (Exception e) {
                System.out.println("Access error: " + e.getMessage());
            }

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}