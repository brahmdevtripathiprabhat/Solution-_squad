// ✅ Toggle Dropdown Visibility
function toggleDropdown() {
    const dropdown = document.getElementById('departmentDropdown');
    dropdown.style.display = (dropdown.style.display === 'none' || dropdown.style.display === '') 
        ? 'block' 
        : 'none';
}

// ✅ Filter by Department
function filterByDepartment(department) {
    const rows = document.querySelectorAll('#dataTable tbody tr');

    rows.forEach(row => {
        const rowDepartment = row.getAttribute('data-department');

        if (department === 'all' || rowDepartment === department) {
            row.style.display = '';   // Show matching rows
        } else {
            row.style.display = 'none';  // Hide non-matching rows
        }
    });

    // Hide dropdown after selection
    document.getElementById('departmentDropdown').style.display = 'none';
}
