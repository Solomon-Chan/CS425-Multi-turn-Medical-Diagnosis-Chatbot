// src/components/Notifications.jsx
import { useEffect, useState } from "react";
import axios from "axios";

export default function Notifications({ accountId }) {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    const fetchNotifications = async () => {
      const res = await axios.get(`http://localhost:8000/notifications/${accountId}`);
      setNotifications(res.data);
    };

    fetchNotifications();

    // poll every 5 seconds
    const interval = setInterval(fetchNotifications, 5000);
    return () => clearInterval(interval);
  }, [accountId]);

  return (
    <div className="p-4 bg-white shadow rounded-lg">
      <h2 className="text-xl font-semibold mb-2">Notifications</h2>
      <ul className="list-disc pl-4">
        {notifications.map((n, i) => (
          <li key={i}>{n.message}</li>
        ))}
      </ul>
    </div>
  );
}
