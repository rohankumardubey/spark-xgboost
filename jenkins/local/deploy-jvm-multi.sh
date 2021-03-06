#!/bin/bash
##
#
# Script to deploy xgboost jar files along with doc and sources jar,
#
# Argument(s):
#   SIGN_FILE: true/false, whether to sign the jar/pom file to de deployed
#
# Used environment(s):
#   OUT:            The absolute path where jar files are placed.
#   SERVER_ID:      The repository id for this deployment.
#   SERVER_URL:     The url where to deploy artifacts.
#   GPG_PASSPHRASE: The passphrase used to sign files, only required when <SIGN_FILE> is true.
###
set -e

echo "Sign Jar?: $1"
SIGN_FILE=$1
###### Build deploy command ######
if [ "$SIGN_FILE" == true ]; then
    DEPLOY_CMD="mvn gpg:sign-and-deploy-file -s settings.xml -Dgpg.passphrase=$GPG_PASSPHRASE"
else
    DEPLOY_CMD="mvn deploy:deploy-file -s settings.xml"
fi
DEPLOY_CMD="$DEPLOY_CMD -Durl=$SERVER_URL -DrepositoryId=$SERVER_ID -B \
    -B -Dmaven.repo.local=$WORKSPACE/.m2"
echo "Deploy cmd: $DEPLOY_CMD"

cd jvm-packages

###### Build jar information ######
REL_VERSION=`mvn exec:exec -q --non-recursive \
    -Dexec.executable=echo \
    -Dexec.args='${project.version}' \
    -Dmaven.repo.local=$WORKSPACE/.m2`
echo "XGBoost version = $REL_VERSION"

###### Start deployment ######
deploySubModule()
{
    FPATH="$OUT/$1/$1_3.0-$REL_VERSION"
    if [ "$SIGN_FILE" == true ]; then
        SRC_DOC_JARS="-Dsources=$FPATH-sources.jar -Djavadoc=$FPATH-javadoc.jar"
    fi
    $DEPLOY_CMD -Dfile=$FPATH.jar -DpomFile=$1/pom.xml $SRC_DOC_JARS
}

# 1) Deploy parent pom file of jvm-packages
$DEPLOY_CMD -Dfile=./pom.xml -DpomFile=./pom.xml

# 2) Deploy sub modules
deploySubModule xgboost4j
deploySubModule xgboost4j-spark

