from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.goto("http://localhost:3000/free-trial")

        # Check if header exists
        header = page.locator("header")
        if header.count() > 0:
            print("Header found!")
        else:
            print("Header NOT found!")

        # Wait for content
        page.wait_for_timeout(1000)

        # Take screenshot of top of page
        page.screenshot(path="verification/free-trial-header.png")

        browser.close()

if __name__ == "__main__":
    run()
