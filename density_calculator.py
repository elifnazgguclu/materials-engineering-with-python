
def calculate_density(mass, volume):
    if mass < 0:
        raise ValueError("Mass cannot be negative.")
    if volume <= 0:
        raise ValueError("Volume must be greater than zero.")

    return mass / volume


try:
    mass = float(input("Enter mass (g): "))
    volume = float(input("Enter volume (cm^3): "))

    density = calculate_density(mass, volume)

    print(f"Density: {density:.3f} g/cm^3")

except ValueError as error:
    print(f"Invalid input: {error}")
