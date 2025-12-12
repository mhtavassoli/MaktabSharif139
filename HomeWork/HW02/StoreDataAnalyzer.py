import os
from datetime import datetime

def load_sales(filename=r"E:\Maktab\Artificial Intelligence\VsCodeExplorer\HomeWork\HW02\sales.txt"):
    """Load sales data from file"""
    sales_data = []
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Empty line
                        continue
                    
                    try:
                        # Split line into parts
                        parts = line.split(',')
                        if len(parts) != 3:
                            continue
                            
                        product_name = parts[0].strip()
                        price = float(parts[1].strip())
                        quantity = int(parts[2].strip())
                        
                        sales_data.append({
                            'product_name': product_name,
                            'price': price,
                            'quantity': quantity
                        })
                        
                    except (ValueError, IndexError):
                        # Skip lines with format errors - no error message
                        continue
                        
    except Exception:
        # General error - no error message
        pass
        
    return sales_data

def save_sale(product_name, price, quantity, filename=r"E:\Maktab\Artificial Intelligence\VsCodeExplorer\HomeWork\HW02\sales.txt"):
    """Save new sale to file"""
    try:
        with open(filename, "a", encoding="utf-8") as file:
            file.write(f"{product_name}, {price}, {quantity}\n")
        return True
    except Exception:
        return False

def calculate_statistics(sales_data):
    """Calculate sales statistics and reports"""
    if not sales_data:
        return {
            'total_transactions': 0,
            'total_sales': 0,
            'top_product': {'name': 'None', 'quantity': 0},
            'average_purchase': 0
        }
    
    # Calculate total sales and number of transactions
    total_sales = 0
    total_transactions = len(sales_data)
    
    # Calculate sales per product
    product_sales = {}
    
    for sale in sales_data:
        sale_amount = sale['price'] * sale['quantity']
        total_sales += sale_amount
        
        # Calculate sales for each product
        product_name = sale['product_name']
        if product_name in product_sales:
            product_sales[product_name] += sale['quantity']
        else:
            product_sales[product_name] = sale['quantity']
    
    # Find best-selling product
    top_product_name = "None"
    top_product_quantity = 0
    
    if product_sales:
        top_product_name = max(product_sales, key=product_sales.get)
        top_product_quantity = product_sales[top_product_name]
    
    # Calculate average purchase amount
    average_purchase = total_sales / total_transactions if total_transactions > 0 else 0
    
    return {
        'total_transactions': total_transactions,
        'total_sales': round(total_sales, 2),
        'top_product': {
            'name': top_product_name,
            'quantity': top_product_quantity
        },
        'average_purchase': round(average_purchase, 2)
    }

def display_report(statistics):
    """Display sales report"""
    print("\n" + "--- Daily Sales Report ---")
    print(f"Total Transactions: {statistics['total_transactions']}")
    print(f"Total Sales: ${statistics['total_sales']}")
    print(f"Best-selling Product: {statistics['top_product']['name']} (Sold {statistics['top_product']['quantity']} items)")
    print(f"Average Purchase Amount: ${statistics['average_purchase']}")
    print("-" * 30)

def add_new_sale():
    """Add new sale"""
    print("\n➕ Add New Sale")
    print("=" * 25)
   
    try:
        product_name = input("Product Name: ").strip()
        if not product_name:
            print("❌ Product name cannot be empty!")
            return
        
        price = float(input("Price: "))
        if price <= 0:
            print("❌ Price must be greater than zero!")
            return
        
        quantity = int(input("Quantity: "))
        if quantity <= 0:
            print("❌ Quantity must be greater than zero!")
            return
        
        if save_sale(product_name, price, quantity):
            print("✅ New sale successfully saved!")
        else:
            print("❌ Error saving new sale!")
            
    except ValueError:
        print("❌ Price and quantity must be numbers!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def show_menu():
    """Display main menu"""
    print("\n" + "=" * 40)
    print("🏪 Store Sales Analysis System")
    print("=" * 40)
    print("1. View Report")
    print("2. Add New Sale")
    print("3. Exit")
    print("=" * 40)

def main():
    """Main program function"""
    print("🎯 Store Sales Analysis System - Started")
    
    while True:
        show_menu()
        
        try:
            choice = input("Please select an option (1-3): ").strip()
            
            if choice == "1":
                # Load data and display report
                sales_data = load_sales()
                statistics = calculate_statistics(sales_data)
                display_report(statistics)
                
            elif choice == "2":
                # Add new sale
                add_new_sale()
                
            elif choice == "3":
                print("\n👋 Thank you for using our system! Goodbye!")
                break
                
            else:
                print("❌ Invalid option! Please enter a number between 1-3.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program stopped by user!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()