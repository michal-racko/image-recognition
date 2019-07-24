# laser_tag

A library made for Arduino-based death tower to be user in a laser tag arena.

Uses video input to identify players of a given team, 
find the closest one and direct the laser towards him.
Moves the camera in the entire azimuthal range if no players found.

Provides a simple GUI to start and select the target team. 

Config options:

`n_steps`   - number of servo-motor steps done in one go

`preview`:  - preview options

`show`          - show video on the monitor

`target`        - show targets found

`players`       - show players found

`target_vector` - show direction to the closest target 

`points`        - show points corresponding to player's LEDs

`groups`:   - player group definitions

            `<group_name>`:
            
`min_hsv`:  - min HSV value for the group color
                
`max_hsv`:  - max HSV value for the group color
