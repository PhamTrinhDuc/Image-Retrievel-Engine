1. Code frontend và backend thành 2 folder riêng, nếu có thể thì code backend thành các microservice
2. Viết logs rõ ràng (json), có trace_id để theo dõi
3. Chia thành các nhánh để code
 - main: nhanh production
 - các nhánh feature khác
 - Chỉ merge vào main qua CI hoặc PR
 - Chia dev/staging/production rõ ràng, tránh việc code không được linh hoạt (rất lưu ý vì làm không tốt sẽ gây rắc rối khi code/test giữa 2 môi trường: dev và prod)
4. Để các biến trong .env 
 - Để các biến như username, password vào .env 
 - Các biến như host, port cũng để vào .env. Lưu ý host sẽ thay đổi theo môi trường (localhost, container)

5. Viết test, triển khai CI sau 1 khoảng thời gian triển khai dự án (CI đơn giản) 
6. Viết api rõ ràng, lỗi thì không được trả ra 200
