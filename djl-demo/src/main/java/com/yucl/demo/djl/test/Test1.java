package com.yucl.demo.djl.test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Test1 {
    public static void main(String[] args) {
        List<Order> orderList = new ArrayList<>();
        orderList.stream().filter(o -> o.getStatus()).collect(Collectors.toList());
    }

    static class Order {
        private boolean status;

        public boolean getStatus() {
            return status;
        }

    }

}
