#!/bin/bash

# Function to capture the last 50 lines of each screen session
capture_screen_logs() {
    # List all active screen sessions
    screen_sessions=$(screen -ls | awk '/\t/ {print $1}')

    if [ -z "$screen_sessions" ]; then
        echo "No active screen sessions found."
        return 1
    fi

    echo "Capturing logs from active screen sessions..."

    # Loop through each screen session
    for session in $screen_sessions; do
        echo "---------------------------------------------------------------------------"
        echo "---------------------------------------------------------------------------"
        echo -e "\nSession: $session"

        # Create a temporary file to store the output
        tmpfile=$(mktemp)

        # Attach to the screen session, dump last 50 lines, and detach
        screen -S "$session" -X hardcopy "$tmpfile"

        if [ -f "$tmpfile" ]; then
            # Extract the last 50 lines of the session
            tail -n 100 "$tmpfile"

            # Clean up the temporary file
            rm -f "$tmpfile"
        else
            echo "Failed to retrieve output for session: $session"
        fi

        # Flush stdout to ensure output is printed immediately

        echo "Pausing for 5 seconds before moving to the next session..." >&2
        sleep 1
    done
}

# Main function
main() {
    capture_screen_logs
}

# Execute main function
main