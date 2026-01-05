BASHRC="$HOME/.bashrc"

# 1. Add or update "running" alias
if ! grep -q "^alias running=" "$BASHRC"; then
  echo "alias running='bjobs -u all -o \"user\" -r | sort | uniq -c | sort -n'" >> "$BASHRC"
  echo "✅ Alias 'running' added to ~/.bashrc"
else
  echo "⚠️  Alias 'running' already exists in ~/.bashrc"
fi

# 2. Add or update "pending" function (formerly called shame)
if ! grep -q "^pending()" "$BASHRC"; then
  cat <<EOF >> "$BASHRC"

# Interactive pending command: shows pending jobs, then prompts to shame top user
pending() {
  echo "== Pending Jobs by User =="
  bjobs -u all -o "user" -p | sort | uniq -c | sort -n
  echo
  read -p "Shame the top user? [y/N] " confirm
  if [[ \$confirm =~ ^[Yy]$ ]]; then
    bjobs -u all -o "user" -p | sort | uniq -c | sort -nr | head -n1 | awk '{print \$2}' | xargs show_user_info
  else
    echo "Shame canceled."
  fi
}
EOF
  echo "✅ Function 'pending' added to ~/.bashrc"
else
  echo "⚠️  Function 'pending' already exists in ~/.bashrc"
fi

# 3. Optionally remove old shame alias or function
if grep -q "^alias shame=" "$BASHRC" || grep -q "^shame()" "$BASHRC"; then
  echo "⚠️  Consider removing the old 'shame' alias/function manually if it's no longer needed."
fi
