package com.yucl.demo.spring.ai.graph.nebula;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.vesoft.nebula.client.graph.SessionPool;
import com.vesoft.nebula.client.graph.SessionPoolConfig;
import com.vesoft.nebula.client.graph.data.HostAddress;
import com.vesoft.nebula.client.graph.data.ResultSet;
import com.vesoft.nebula.client.graph.exception.AuthFailedException;
import com.vesoft.nebula.client.graph.exception.BindSpaceFailedException;
import com.vesoft.nebula.client.graph.exception.ClientServerIncompatibleException;
import com.vesoft.nebula.client.graph.exception.IOErrorException;


@Service
public class NebulaClient {
    private Logger logger = LoggerFactory.getLogger(NebulaClient.class);

    private SessionPool sessionPool = createSessonPool();

    private static final String rel_query = 
                "MATCH ()-[e:`$edge_type`]->()\n" + 
                "  WITH e limit 1\n" + 
                "MATCH (m)-[:`$edge_type`]->(n) WHERE id(m) == src(e) AND id(n) == dst(e)\n" + 
                "RETURN \"(:\" + tags(m)[0] + \")-[:$edge_type]->(:\" + tags(n)[0] + \")\" AS rels\n" 
              ;

    public static void main(String[] args) {
        NebulaClient nebulaClient = new NebulaClient();
        try {
            nebulaClient.getAllEdges();
            nebulaClient.getAllTags();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        nebulaClient.sessionPool.close();
    }

    private SessionPool createSessonPool() {
        List<HostAddress> addresses = Arrays.asList(new HostAddress("192.168.32.129", 9669));
        String spaceName = "test";
        String user = "root";
        String password = "nebula";
        SessionPoolConfig sessionPoolConfig = new SessionPoolConfig(addresses, spaceName, user, password);
        // .setMaxSessionSize(10)
        // .setMinSessionSize(10)
        // .setRetryConnectTimes(3)
        // .setWaitTime(100)
        // .setRetryTimes(3)
        // .setIntervalTime(100);
        SessionPool sessionPool = new SessionPool(sessionPoolConfig);
        if (!sessionPool.hasInit.get()) {
            logger.error("session pool init failed.");
            throw new RuntimeException("session pool init failed.");
        }
        return sessionPool;

    }

    public void execute(String stmt) {
        ResultSet resultSet;
        try {
            resultSet = sessionPool.execute(stmt);
            System.out.println(resultSet.toString());

        } catch (IOErrorException | ClientServerIncompatibleException | AuthFailedException
                | BindSpaceFailedException e) {
            e.printStackTrace();
            System.exit(1);

        }
    }

    public void getAllTags() throws Exception {
        ResultSet tagsResult = sessionPool.execute("SHOW TAGS ;");
        printColNames(tagsResult);
        if (tagsResult.isSucceeded()) {
            for (int i = 0; i < tagsResult.rowsSize(); i++) {
                ResultSet.Record tagRecord = tagsResult.rowValues(i);
                String tagName = tagRecord.values().get(0).asString();
                ResultSet tagDescResult = sessionPool.execute("DESCRIBE TAG " + tagName + ";");
                printColNames(tagDescResult);
                if (tagDescResult.isSucceeded()) {
                    for (int j = 0; j < tagDescResult.rowsSize(); j++) {
                        ResultSet.Record tagDescRecord = tagDescResult.rowValues(j);
                        System.out.println("Field Name: " + tagDescRecord.values().get(0)); // 属性名
                        System.out.println("Type: " + tagDescRecord.values().get(1)); // 属性类型
                        System.out.println("Default Value: " + tagDescRecord.values().get(2)); // 默认值
                        System.out.println("Comment: " + tagDescRecord.values().get(4)); // 默认值

                    }
                }

            }
        } else {
            logger.error("Failed to show edges: " + tagsResult.getErrorMessage());
        }
    }

    public void getAllEdges() throws Exception {
        ResultSet edgesResult = sessionPool.execute("SHOW EDGES;");
        printColNames(edgesResult);
        if (edgesResult.isSucceeded()) {
            for (int i = 0; i < edgesResult.rowsSize(); i++) {
                ResultSet.Record edgeRecord = edgesResult.rowValues(i);
                String edgeName = edgeRecord.values().get(0).asString();
                String comment = getComment(edgeName);
                System.out.println(comment);
                ResultSet edgeDescResult = sessionPool.execute("DESCRIBE EDGE " + edgeName + ";");
                printColNames(edgeDescResult);
                if (edgeDescResult.isSucceeded()) {
                    for (int j = 0; j < edgeDescResult.rowsSize(); j++) {
                        ResultSet.Record edgeDescRecord = edgeDescResult.rowValues(j);
                        System.out.println("Field Name: " + edgeDescRecord.values().get(0)); // 属性名
                        System.out.println("Type: " + edgeDescRecord.values().get(1)); // 属性类型
                        System.out.println("Default Value: " + edgeDescRecord.values().get(2)); // 默认值
                        System.out.println("Comment: " + edgeDescRecord.values().get(4)); // 默认值

                    }
                }

            }
        } else {
            logger.error("Failed to show edges: " + edgesResult.getErrorMessage());
        }
    }

    private String getComment(String edgeName) throws Exception {
        ResultSet tagCreateResult = sessionPool.execute("SHOW CREATE EDGE " + edgeName + ";");
        String comment = "";
        if (tagCreateResult.isSucceeded()) {
            ResultSet.Record tagCreateRecord = tagCreateResult.rowValues(0);
            String createTagStmt = tagCreateRecord.values().get(1).asString();
            System.out.println(createTagStmt);           
            String regex = "comment\\s*=\\s*\"([^\"]*)\"";
            Pattern pattern = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
            Matcher matcher = pattern.matcher(createTagStmt);
            if (matcher.find()) {
                System.out.println("Comment value: " + matcher.group(1));
            } else {
                System.out.println("No match found.");
            }
        }
        return comment;

    }

    private void printColNames(ResultSet result) {
        List<String> colNames = result.keys();
        for (String name : colNames) {
            System.out.printf("%15s |", name);
        }
        System.out.println();
    }

}
